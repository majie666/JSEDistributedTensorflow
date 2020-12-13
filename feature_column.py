import tensorflow as tf
from tensorflow import feature_column

"""--------------------------------------------- Dense column -------------------------------------------------"""

"""numeric column"""
def transform_fn(x):
    return x + 2

with tf.Session() as sess:
    price = {'price': [[1.], [2.], [3.], [4.]]}
    price_column = feature_column.numeric_column('price', normalizer_fn=transform_fn)
    price_transformed_tensor = feature_column.input_layer(price, [price_column])
    print(sess.run([price_transformed_tensor]))


"""embedding_column"""
with tf.Session() as sess:
    color_data = {'color': [['R', 'G'], ['G', 'A'], ['B', 'B'], ['A', 'A']]}
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )
    color_embeding = feature_column.embedding_column(color_column, 4, combiner='sum')
    color_embeding_dense_tensor = feature_column.input_layer(color_data, [color_embeding])
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run([color_embeding_dense_tensor]))



"""--------------------------------------------- bucketized_column -------------------------------------------------"""

"""bucketized_column"""
with tf.Session() as session:
    price = {'price': [[5.], [15.], [25.], [35.]]}
    price_column = feature_column.numeric_column('price')
    bucket_price = feature_column.bucketized_column(price_column, [10, 20, 30, 40])
    price_bucket_tensor = feature_column.input_layer(price, [bucket_price])
    print(session.run([price_bucket_tensor]))



"""--------------------------------------------- Categorical_column -------------------------------------------------"""

"""categorical_column_with_identity"""
with tf.Session() as sess:
    color_data = {'color': [[2], [5], [-1], [0]]}
    color_column = feature_column.categorical_column_with_identity('color', 7)
    color_column_identy = feature_column.indicator_column(color_column)
    color_dense_tensor = feature_column.input_layer(color_data, [color_column_identy])
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run([color_dense_tensor]))


"""categorical_column_with_vocabulary_list"""
with tf.Session() as sess:
    color_data = {'color': [['R', 'R'], ['G', 'R'], ['B', 'G'], ['A', 'A']]}
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )
    color_column_identy = feature_column.indicator_column(color_column)
    color_dense_tensor = feature_column.input_layer(color_data, [color_column_identy])
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
print(sess.run([color_dense_tensor]))


"""categorical_column_with_hash_bucket"""
with tf.Session() as sess:
    color_data = {'color': [[2], [5], [-1], [0]]}
    color_column = feature_column.categorical_column_with_hash_bucket('color', 7, dtype=tf.int32)
    color_column_identy = feature_column.indicator_column(color_column)
    color_dense_tensor = feature_column.input_layer(color_data, [color_column_identy])
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
print(sess.run([color_dense_tensor]))


"""crossed_column"""
with tf.Session() as sess:
    featrues = {
        'price': [['A'], ['B'], ['C']],
        'color': [['R'], ['G'], ['B']]
    }
    price = feature_column.categorical_column_with_vocabulary_list('price', ['A', 'B', 'C', 'D'])
    color = feature_column.categorical_column_with_vocabulary_list('color', ['R', 'G', 'B'])
    p_x_c = feature_column.crossed_column([price, color], 16)
    p_x_c_identy = feature_column.indicator_column(p_x_c)
    p_x_c_identy_dense_tensor = feature_column.input_layer(featrues, [p_x_c_identy])
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run([p_x_c_identy_dense_tensor]))



"""--------------------------------------------- Senior_column -------------------------------------------------"""

"""shared_embedding_columns"""
with tf.Session() as sess:
    color_data = {'color': [[2, 2], [5, 5], [0, -1], [0, 0]],
                  'color2': [[2], [5], [-1], [0]]}  # 4行样本
    color_column = feature_column.categorical_column_with_hash_bucket('color', 7, dtype=tf.int32)
    color_column2 = feature_column.categorical_column_with_hash_bucket('color2', 7, dtype=tf.int32)
    color_column_embed = feature_column.shared_embedding_columns([color_column2, color_column], 3, combiner='sum')
    print(type(color_column_embed))
    color_dense_tensor = feature_column.input_layer(color_data, color_column_embed)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run(color_dense_tensor))


"""weighted_categorical_column"""
features = {'color': [['R'], ['A'], ['G'], ['B'], ['R']],
            'weight': [[1.0], [5.0], [4.0], [8.0], [3.0]]}

color_f_c = tf.feature_column.categorical_column_with_vocabulary_list(
    'color', ['R', 'G', 'B', 'A'], dtype=tf.string, default_value=-1
)

column = tf.feature_column.weighted_categorical_column(color_f_c, 'weight')
indicator = tf.feature_column.indicator_column(column)
tensor = tf.feature_column.input_layer(features, [indicator])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([tensor]))


"""embedding_lookup"""
import tensorflow as tf
import numpy as np

a = [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3], [4.1, 4.2, 4.3]]
a = np.asarray(a)
idx1 = tf.Variable([0, 2, 3, 1], tf.int32)
idx2 = tf.Variable([[0, 2, 3, 1], [4, 0, 2, 2]], tf.int32)
out1 = tf.nn.embedding_lookup(a, idx1)
out2 = tf.nn.embedding_lookup(a, idx2)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(out1))
    print('==================')
    print(sess.run(out2))


"""embedding_lookup"""
# 如果想要根据embedding_lookup准确找出每个feature对应的embedding，则必须将feature转化为label_encoder
color_data = {'color': [[1], [2], [3], [4]]}
color_column = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_identity('color',5),4)
color_dense_tensor = feature_column.input_layer(color_data , [color_column])

idx1 = tf.Variable([1,3,2], tf.string)
# idx1 = tf.string_to_hash_bucket_fast(idx1,4)
out1 = tf.nn.embedding_lookup(color_dense_tensor, idx1)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(idx1))
    print(sess.run(color_dense_tensor))
    print(sess.run(out1))
    print('==================')


"""多值离散特征的embedding表示"""
with tf.Session() as sess:
    # 定义所有商品的集合
    good_sets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    # 假设现在有两个用户，点击了a,b,c,d 和 a,e,f

    click_data = {'click': ["a,b,c,d","a,e,f"]}

    color_column = feature_column.categorical_column_with_vocabulary_list(
        'click', good_sets, dtype=tf.string, default_value=-1
    )

    embedding_column = feature_column.embedding_column(color_column, 3, combiner='mean')

    click_data['click'] = tf.string_split(click_data['click'] ,delimiter=',')

    column_input_layer = feature_column.input_layer(click_data, [embedding_column])
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run([click_data]))
    print(sess.run([column_input_layer]))


"---------------------------------"
# embedding_user_user_id_profile_name = ['embedding_user_user_id', 'embedding_user_group', 'embedding_user_cluster', \
#                                    'embedding_user_gender', 'embedding_user_age', 'embedding_user_has_child',
#                                    'embedding_user_baby_gender', \
#                                    'embedding_user_baby_age', 'embedding_user_is_marr', 'embedding_user_price_level',
#                                    'embedding_user_discount_motive', \
#                                    'embedding_user_province', 'embedding_user_city', 'embedding_user_county']
# embedding_user_profile_size = [-1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
# W = {}
# for i in range(1, len(embedding_user_profile_name)):
#     W[embedding_user_profile_name[i]] = tf.get_embedding_variable(embedding_user_profile_name[i],
#                                                                   embedding_dim=embedding_user_profile_size[i],
#                                                                   # key_dtype=tf.string,
#                                                                   initializer=tf.random_uniform_initializer(
#                                                                       minval=-1.0, maxval=1.0),
#                                                                   partitioner=tf.fixed_size_partitioner(
#                                                                       num_shards=4))
#
# n_input_user_profile = 0
# for i in range(1, len(embedding_user_profile_name)):
#     if i == 1:
#         user_profile_embed_batch = tf.nn.embedding_lookup(W[embedding_user_profile_name[i]],
#                                                           string_ops.string_to_hash64(user_profile_batch[:, i]))
#     else:
#         embed_tmp = tf.nn.embedding_lookup(W[embedding_user_profile_name[i]],
#                                            string_ops.string_to_hash64(user_profile_batch[:, i]))
#         user_profile_embed_batch = tf.concat([user_profile_embed_batch, embed_tmp], 1)
#     n_input_user_profile = n_input_user_profile + embedding_user_profile_size[i]
#
# W[embedding_item_profile_name[0]] = tf.get_variable(embedding_item_profile_name[0], \
#                                                     initializer=tf.random_uniform(
#                                                         [target_num, embedding_item_profile_size[0]], -1.0, 1.0,
#                                                         dtype=tf.float32))



