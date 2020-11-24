import tensorflow as tf
from tensorflow import feature_column

"""
numeric_column

[[3.]
 [4.]
 [5.]
 [6.]]
"""
# def transform_fn(x):
#     return x + 2
#
# with tf.Session() as sess:
#     price = {'price':[[1.],[2.],[3.],[4.]]}
#     price_column = feature_column.numeric_column('price',normalizer_fn=transform_fn)
#     price_transformed_tensor = feature_column.input_layer(price,[price_column])
#     print(sess.run(price_transformed_tensor))


"""
embedding_column

[array([[ 1.1850209 , -0.06238219, -0.6394337 , -0.3073619 ],
       [ 0.9771619 ,  0.47652134, -0.42376053,  0.33622897],
       [ 0.7381851 , -0.18736736,  0.2083951 , -0.820421  ],
       [ 0.        ,  0.        ,  0.        ,  0.        ]],
      dtype=float32)]
"""
# with tf.Session() as sess:
#     color_data = {'color':[['R','G'],['G','A'],['B','B'],['A','A']]}
#     color_column = feature_column.categorical_column_with_vocabulary_list(
#         'color',['R','G','B'],dtype=tf.string,default_value=-1
#     )
#     color_embedding = feature_column.embedding_column(color_column,4,combiner='sum')
#     color_embedding_tensor = feature_column.input_layer(color_data,[color_embedding])
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.tables_initializer())
#     print(sess.run([color_embedding_tensor]))


"""
bucketized_column

[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]]
"""
with tf.Session() as sess:
    price = {'price': [[5.], [15.], [20.], [35.]]}
    price_column = feature_column.numeric_column('price')
    bucket_price = feature_column.bucketized_column(price_column,[10,20,30,40])
    price_bucket_tensor = feature_column.input_layer(price,[bucket_price])
    print(sess.run(price_bucket_tensor))


