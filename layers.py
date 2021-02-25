import tensorflow as tf


def dnn_layer(dnn_input, l2_reg=0, batch_norm=True, units=[16, 8], is_training=False, return_type="layer"):
    """
    dnn 组件
    dnn_input:(-1, n)
    """
    dnn_layer = dnn_input
    for i in range(len(units)):
        tmp_dnn_layer = tf.contrib.layers.fully_connected(dnn_layer, units[i], activation_fn=tf.nn.relu,
                                                          weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                              scale=l2_reg))
        tmp_dnn_layer = tf.contrib.layers.batch_norm(tmp_dnn_layer,
                                                     is_training=is_training) if batch_norm else tmp_dnn_layer
        dnn_layer = tmp_dnn_layer
    if return_type == "logits":
        dnn_logits = tf.contrib.layers.fully_connected(dnn_layer, 1, activation_fn=None)
        return dnn_logits
    return dnn_layer


def fm_layer(fm_input_list):
    """
    fm 组件
    fm_input_list : list, (-1, emb_size)
    """
    sum_square = tf.square(tf.reduce_sum(fm_input_list, axis=0))
    square_sum = tf.reduce_sum(tf.square(fm_input_list), axis=0)
    fm_logits = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1)
    return fm_logits


def lr_layer(lr_input, l2_reg=0):
    """
    LR 组件
    lr_input: (-1, n)
    """
    print(lr_input)
    tf.contrib.layers.apply_regularization(
        tf.contrib.layers.l2_regularizer(l2_reg),
        weights_list=[lr_input]
    )
    lr_logits = tf.reduce_sum(lr_input, axis=1)
    return lr_logits


"""
这里的输入有三个，候选广告queries，用户历史行为keys，以及Batch中每个行为的长度。这里为什么要输入一个keys_length呢，因为每个用户发生过的历史行为是不一样多的，
但是输入的keys维度是固定的(都是历史行为最大的长度)，因此我们需要这个长度来计算一个mask，告诉模型哪些行为是没用的，哪些是用来计算用户兴趣分布的。

将queries变为和keys同样的形状B * T * H(B指batch的大小，T指用户历史行为的最大长度，H指embedding的长度)
通过三层神经网络得到queries和keys中每个key的权重，并经过softmax进行标准化
通过weighted sum得到最终用户的历史行为分布
"""


def attention(queries, keys, keys_length):
    """
        queries:     [B, H]
        keys:        [B, T, H]
        keys_length: [B]
    """

    queries_hidden_units = queries.get_shape().as_list()[-1]
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])

    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  # B*T*4H
    # 三层全链接
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att')  # B*T*1

    outputs = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])  # B*1*T
    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])
    key_masks = tf.expand_dims(key_masks, 1)  # B*1*T
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # 在补足的地方附上一个很小的值，而不是0
    outputs = tf.where(key_masks, outputs, paddings)  # B * 1 * T
    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
    # Activation
    outputs = tf.nn.softmax(outputs)  # B * 1 * T
    # Weighted Sum
    outputs = tf.matmul(outputs, keys)  # B * 1 * H 三维矩阵相乘，相乘发生在后两维，即 B * (( 1 * T ) * ( T * H ))
    return outputs


def attention_layer(
        key, value, true_length, emb_size, sequence_length,
        l2_reg, batch_norm=True, units=[16, 8], is_training=False):
    """
    attention 组件
    key:(-1, emb_size)
    val:(-1, sequence_length * emb_size)
    输出:(-1, emb_size)
    """
    # key输入处理
    key_input = tf.tile(key, multiples=[1, sequence_length])
    key_input = tf.reshape(key_input, [-1, emb_size])
    # value输入处理
    value_input = tf.reshape(value, [-1, emb_size])
    # 外积
    out_product_input1 = tf.reshape(value_input, [-1, emb_size, 1])
    out_product_input2 = tf.reshape(key_input, [-1, 1, emb_size])
    out_product = tf.matmul(out_product_input1, out_product_input2)
    out_product = tf.reshape(out_product, [-1, emb_size * emb_size])
    # attention
    attention_input = tf.concat([key_input, out_product, value_input], 1)
    activation_dnn = dnn_layer(attention_input, l2_reg=l2_reg, batch_norm=batch_norm, units=units,
                               is_training=is_training)
    activation = tf.contrib.layers.fully_connected(activation_dnn, 1, activation_fn=None)
    activation = tf.reshape(activation, [-1, sequence_length])
    # 组合attention结果
    activation_mask = tf.sequence_mask(true_length, sequence_length)
    activation_mask = tf.reshape(activation_mask, [-1, sequence_length])
    activation = tf.where(activation_mask, activation, tf.ones_like(activation) * -1e6)
    activation = tf.nn.softmax(activation)
    result = tf.reshape(activation, [-1, 1]) * value_input
    result = tf.reshape(result, [-1, sequence_length, emb_size])
    result = tf.reduce_sum(result, 1)
    return result


def sequence_avg_pooling_layer(sequence_input, true_length, emb_size, sequence_length):
    """
    取序列各embedding的均值
    sequence_input (-1, emb_size * sequence_length)
    输出:(-1, emb_size)
    """
    sequence_mask = tf.sequence_mask(true_length, sequence_length)
    sequence_mask = tf.tile(tf.reshape(sequence_mask, [-1, 1]), multiples=[1, emb_size])
    sequence_input = tf.reshape(sequence_input, [-1, emb_size])
    sequence_input = tf.where(sequence_mask, sequence_input, tf.zeros_like(sequence_input))
    sequence_input = tf.reshape(sequence_input, [-1, sequence_length, emb_size])
    avg_pooling_result = tf.reduce_sum(sequence_input, axis=1) / (true_length + 1e-10)
    return avg_pooling_result


def self_attention_layer(
        sequence_input, true_length, sequence_length,
        input_emb_size, output_emb_size, l2_reg=0.0):
    """
    self attention
    sequence_input (-1, input_emb_size * sequence_length)
    输出:(-1, output_emb_size * sequence_length)
    """
    sequence_input = tf.reshape(sequence_input, [-1, input_emb_size])
    Q_emb = tf.contrib.layers.fully_connected(
        sequence_input, output_emb_size,
        activation_fn=None, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg)
    )
    K_emb = tf.contrib.layers.fully_connected(
        sequence_input, output_emb_size,
        activation_fn=None, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg)
    )
    V_emb = tf.contrib.layers.fully_connected(
        sequence_input, output_emb_size,
        activation_fn=None, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg)
    )
    Q_emb = tf.reshape(Q_emb, [-1, sequence_length, output_emb_size])
    K_emb = tf.reshape(K_emb, [-1, sequence_length, output_emb_size])
    V_emb = tf.reshape(V_emb, [-1, sequence_length, output_emb_size])
    V_weight = tf.matmul(Q_emb, tf.transpose(K_emb, [0, 2, 1]))
    V_weight = tf.nn.softmax(V_weight)
    result_emb = tf.matmul(V_weight, V_emb)
    result_emb = tf.reshape(result_emb, [-1, sequence_length * output_emb_size])
    return result_emb


def multi_head_attention_layer(
        sequence_input, true_length, sequence_length,
        input_emb_size, output_emb_size, num_heads, l2_reg=0.0, self_attention_emb_size=None):
    """
    multi head attention
    sequence_input (-1, input_emb_size * sequence_length)
    输出:(-1, output_emb_size * sequence_length)
    """
    self_attention_emb_size = output_emb_size if self_attention_emb_size is None else self_attention_emb_size
    self_attention_emb_list = [
        self_attention_layer(
            sequence_input=sequence_input, true_length=true_length, sequence_length=sequence_length,
            input_emb_size=input_emb_size, output_emb_size=self_attention_emb_size, l2_reg=l2_reg
        ) for _ in range(num_heads)
    ]
    self_attention_emb_list = [
        tf.reshape(emb, [-1, self_attention_emb_size])
        for emb in self_attention_emb_list
    ]
    input_emb = tf.concat(self_attention_emb_list, 1)
    result_emb = tf.contrib.layers.fully_connected(
        input_emb, output_emb_size, activation_fn=tf.nn.relu,
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg)
    )
    result_emb = tf.reshape(result_emb, [-1, sequence_length * output_emb_size])
    return result_emb


def transformer_layer(
        sequence_input, true_length, sequence_length,
        input_emb_size, output_emb_size, blocks, num_heads,
        l2_reg, batch_norm=True, is_training=False, self_attention_emb_size=None):
    """
    multi head attention
    sequence_input (-1, input_emb_size * sequence_length)
    输出:(-1, output_emb_size)
    """
    self_attention_emb_size = input_emb_size if self_attention_emb_size is None else self_attention_emb_size
    tmp_sequence_input = tf.reshape(sequence_input, [-1, input_emb_size * sequence_length])
    for _ in range(blocks):
        # multi-head attention
        multi_head_attention_emb = multi_head_attention_layer(
            sequence_input=tmp_sequence_input, true_length=true_length, sequence_length=sequence_length,
            input_emb_size=input_emb_size, output_emb_size=input_emb_size,
            num_heads=num_heads, l2_reg=l2_reg, self_attention_emb_size=self_attention_emb_size
        )
        # add & norm
        fnn_input = tmp_sequence_input + multi_head_attention_emb
        fnn_input = tf.contrib.layers.batch_norm(fnn_input, is_training=is_training) if batch_norm else fnn_input
        # feed forward
        fnn_input = tf.reshape(fnn_input, [-1, input_emb_size])
        fnn_output = dnn_layer(fnn_input, l2_reg=l2_reg, batch_norm=False, units=[input_emb_size, input_emb_size])
        # add & norm
        blocks_output = fnn_output + fnn_input
        blocks_output = tf.reshape(blocks_output, [-1, input_emb_size * sequence_length])
        blocks_output = tf.contrib.layers.batch_norm(blocks_output,
                                                     is_training=is_training) if batch_norm else blocks_output
        # 记录当前循环结果
        tmp_sequence_input = blocks_output
    # 把最后一次的输出结果输入到全连接层
    transformer_result = tf.contrib.layers.fully_connected(
        tmp_sequence_input, output_emb_size, activation_fn=tf.nn.relu,
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg)
    )
    return transformer_result
