import numpy as np
from scipy.sparse import csr_matrix

def transform_indices(data, users, items):
    data_index = {}
    for entity, field in zip(['users', 'items'], [users, items]):
        idx, idx_map = to_numeric_id(data, field)
        data_index[entity] = idx_map
        data.loc[:, field] = idx
    return data, data_index

def to_numeric_id(data, field):
    idx_data = data[field].astype("category")
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map


def matrix_from_data(data, data_description, dtype=None):
    '''
    Converts pandas DataFrame into sparse CSR matrix.
    Assumes data in the DataFrame is alread normalized via `transform_indices`.
    '''
    # get indices of observed data
    user_idx = data[data_description['users']].values
    item_idx = data[data_description['items']].values
    feedback_data = data_description.get('feedback', None)
    if feedback_data is not None:
        feedback = data[feedback_data].values
    else:
        feedback = np.ones(len(user_idx))
    # construct rating matrix
    shape = (data_description['n_users'], data_description['n_items'])
    return csr_matrix((feedback, (user_idx, item_idx)), shape=shape, dtype=dtype)