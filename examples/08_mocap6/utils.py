import numpy as np
import bnpy
from hmmlearn.hmm import GaussianHMM
from bnpy import HModel
import time

def get_GroupXData_from_list(features_list):
    features_concat = np.vstack(features_list)
    features_len = [0]
    n_doc = 0
    rolling_total = 0

    for i in range(len(features_list)):
        features_len.append(len(features_list[i]) + rolling_total)
        rolling_total += len(features_list[i])
        n_doc += 1

    features_len = np.array(features_len)

    return bnpy.data.GroupXData(X=features_concat, doc_range=features_len,
                                          nDocTotal=n_doc)
def get_hmm_learn_from_bnpy(some_model: HModel):

    obs_model = some_model.obsModel
    total_k = obs_model.K
    means = []
    sigmas = []
    for k in range(total_k):
        sigmas.append(np.diag(obs_model.get_covar_mat_for_comp(k)))
        means.append(obs_model.get_mean_for_comp(k))

    means = np.vstack(means)
    sigmas = np.vstack(sigmas)

    A = some_model.allocModel.get_trans_prob_matrix(),
    pi = some_model.allocModel.get_init_prob_vector(),

    # creat hmm
    hmm_bnpy = GaussianHMM(n_components=len(pi[0]), covariance_type='diag', init_params='')
    hmm_bnpy.n_features = means.shape[1]
    hmm_bnpy.transmat_, hmm_bnpy.startprob_, hmm_bnpy.means_ = normalize_matrix(A[0]), normalize_matrix(pi[0]), means
    hmm_bnpy.covars_ = sigmas
    return hmm_bnpy

def normalize_matrix(matrix):
    matrix += 1e-40
    return matrix / np.sum(matrix, axis=(matrix.ndim - 1), keepdims=True)

def filter_data_with_labels(some_data, some_labels, label):
    whale_dat_indicis = np.where(some_labels == label)[0]
    filtered_data = []
    for i in range(len(some_data)):
        if i in whale_dat_indicis:
            filtered_data.append(some_data[i])
    return filtered_data

def train_model(model_type, n_components, train_data, train_data_bnpy, val_data, hmm_kwargs_arr):

    trained_model = {}
    start_time = time.time()

    if model_type == 'hmmlearn':
        curr_hmm = GaussianHMM(n_components=n_components, covariance_type='diag')
        curr_hmm.fit(np.concatenate(train_data))
        # save
        trained_model['model'] = curr_hmm
        trained_model['elbo'] = 0
        trained_model['bnpy_model'] = {}
        trained_model['bnpy_hist'] = {}
    elif model_type == 'em_bnpy':
        hmmdiag_trained_model, hmmdiag_info_dict = bnpy.run(
        train_data_bnpy, 'FiniteHMM', 'DiagGauss', 'EM',
        output_path='/tmp/mocap6/showcase-K=20-model=HDPHMM+DiagGauss-ECovMat=1*eye/',
        **dict(
            sum(map(list, [hmm_kwargs.items() for hmm_kwargs in hmm_kwargs_arr]), [])))
        model = get_hmm_learn_from_bnpy(hmmdiag_trained_model)
        # save
        trained_model['model'] = model
        trained_model['elbo'] = hmmdiag_info_dict['loss']
        trained_model['bnpy_model'] = hmmdiag_trained_model
        trained_model['bnpy_hist'] = hmmdiag_info_dict
    elif model_type == 'vi_fin_bnpy':
        hmmdiag_trained_model, hmmdiag_info_dict = bnpy.run(
        train_data_bnpy, 'FiniteHMM', 'DiagGauss', 'memoVB',
        output_path='/tmp/mocap6/showcase-K=20-model=HDPHMM+DiagGauss-ECovMat=1*eye/',
        **dict(
            sum(map(list, [hmm_kwargs.items() for hmm_kwargs in hmm_kwargs_arr]), [])))
        # save
        model = get_hmm_learn_from_bnpy(hmmdiag_trained_model)
        trained_model['model'] = model
        trained_model['elbo'] = hmmdiag_info_dict['loss']
        trained_model['bnpy_model'] = hmmdiag_trained_model
        trained_model['bnpy_hist'] = hmmdiag_info_dict
    elif model_type == 'vi_inf_bnpy':
        hmmdiag_trained_model, hmmdiag_info_dict = bnpy.run(
        train_data_bnpy, 'HDPHMM', 'DiagGauss', 'memoVB',
        output_path='/tmp/mocap6/showcase-K=20-model=HDPHMM+DiagGauss-ECovMat=1*eye/',
        **dict(
            sum(map(list, [hmm_kwargs.items() for hmm_kwargs in hmm_kwargs_arr]), [])))
        # save
        model = get_hmm_learn_from_bnpy(hmmdiag_trained_model)
        trained_model['model'] = model
        trained_model['elbo'] = hmmdiag_info_dict['loss']
        trained_model['bnpy_model'] = hmmdiag_trained_model
        trained_model['bnpy_hist'] = hmmdiag_info_dict

    # val
    trained_model['val'] = trained_model["model"].score(np.concatenate(val_data))
    trained_model['bic'] = trained_model["model"].bic(np.concatenate(val_data))
    trained_model['aic'] = trained_model["model"].aic(np.concatenate(val_data))

    # N comps
    n_zero_comps = np.sum(np.isclose(np.sum(trained_model["model"].means_, axis=1), 0))
    n_comps = trained_model["model"].means_.shape[0]
    trained_model["final_comps"] = n_comps - n_zero_comps

    # Stats
    trained_model["time"] = time.time() - start_time

    return trained_model


def get_all_results(model_types, num_components_to_test, whale_labels, whale_data, test_args, bnpy_kwargs_arr):

    n_inits = test_args['n_inits']
    cv_amt = test_args['cv_amt']

    all_whale_results = {}

    for whale_label in whale_labels:

        print('testing for whale type: ', whale_label)
        all_results = {}

        for model_ind in range(len(model_types)):

            print('testing for model_type: ', model_types[model_ind])
            model_results = {}
            results_per_component = {}  # results per dimension
            start_outer = time.time()
            cv_len = 0

            for num_comps in num_components_to_test:
                print('_______________________________')
                print('whale: ', whale_label,
                      ' mode: ', model_types[model_ind],
                      ' num comps: ', num_comps)
                print('_______________________________')

                cv_len = len(whale_data['train_data'])
                cv_test = whale_data['test_data']

                bnpy_kwargs_arr.append(dict(K=num_comps))
                trained_models = []
                total_inits = 0

                for cv_index in range(cv_amt):

                    train_for_whale = filter_data_with_labels(whale_data['train_data'][cv_index],whale_data['train_label'][cv_index], whale_label)
                    train_for_whale_bnpy = get_GroupXData_from_list(train_for_whale)
                    val_for_whale = filter_data_with_labels(whale_data['val_data'][cv_index],whale_data['val_label'][cv_index], whale_label)

                    for i in n_inits:
                        total_inits += 1
                        model_it = None
                        try:
                            model_it = train_model(
                                model_types[model_ind],
                                num_comps,
                                train_for_whale,
                                train_for_whale_bnpy,
                                val_for_whale,
                                bnpy_kwargs_arr,
                            )

                        # Code that may raise an exception
                        # ...
                        except ValueError as e:
                            print(e)
                        # Code to handle the exception
                        else:
                            trained_models.append(model_it)
                        # Code to be executed if no exception occurs in the try block
                        finally:
                            pass
                            # Code that will be executed no matter what, whether an exception occurs or not
                            #    ...

                best_model_ind = np.argmax([trained_model['val'] for trained_model in trained_models])
                best_model = trained_models[best_model_ind]['model']
                average_score = np.mean([trained_model['val'] for trained_model in trained_models])

                results_per_component[num_comps] = {
                    'lls': [trained_model['val'] for trained_model in trained_models],
                    'elbos': [trained_model['elbo'] for trained_model in trained_models],
                    'models': [trained_model['model'] for trained_model in trained_models],
                    'bnpy_model': [trained_model['bnpy_model'] for trained_model in trained_models],
                    'bnpy_hist': [trained_model['bnpy_hist'] for trained_model in trained_models],
                    'test': best_model.score(np.concatenate(cv_test)),
                    'avg_val': average_score,
                    'final_components': best_model.n_components,
                    'final_components_avg': np.mean([trained_model['model'].n_components for trained_model in trained_models]),
                    'its': total_inits,
                    'time': [trained_model['time'] for trained_model in trained_models],
                    'aic': np.mean([trained_model['aic'] for trained_model in trained_models]),
                    'bic': np.mean([trained_model['bic'] for trained_model in trained_models]),
                    'best_model': best_model
                }

            end_outer = time.time()

            model_results['total_time'] = end_outer - start_outer
            model_results['components'] = results_per_component
            model_results['total_its'] = cv_len * len(n_inits)
            model_results['component_list'] = num_components_to_test

            all_results[model_types[model_ind]] = model_results

        all_whale_results[whale_label] = all_results

    return all_whale_results


