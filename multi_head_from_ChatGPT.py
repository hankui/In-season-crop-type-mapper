import tensorflow as tf

## this is refined from chatGPT 
# self = attention_layer
# v, k, q = values, keys, queries
class MultiHeadAttentionWithRelativePositionBias(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, seq_length):
        super(MultiHeadAttentionWithRelativePositionBias, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.seq_length = seq_length
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
        
        # Initialize learnable relative position bias
        self.relative_position_bias = self.add_weight(
            "relative_position_bias",
            # shape=[2 * seq_length - 1, self.depth],
            shape=[2 * seq_length - 1, num_heads],
            initializer="random_normal",
            trainable=True
        )
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_length, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, position_array, valid_mask):
        """Calculate the attention weights."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_length, seq_length)
        
        # Adjust position indices to be in the range of [0, 2*seq_length-2]
        pos_indices = position_array + self.seq_length - 2
        # valid_mask = tf.not_equal(position_array, filled_value) # need to be changed 
        # pos_indices = tf.where(valid_mask, pos_indices, tf.zeros_like(pos_indices))
        # Gather relative positional encodings
        # A = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)  # shape [2, 3]
        # Expand dimensions to prepare for broadcasting
        A_expanded_i = tf.expand_dims(position_array, axis=1)  # shape [a, b, 1]
        A_expanded_j = tf.expand_dims(position_array, axis=2)  # shape [a, 1, b]
        pos_indices = A_expanded_i - A_expanded_j + (self.seq_length-1) # shape [a, b, b]
        pos_enc = tf.gather(self.relative_position_bias, tf.cast(tf.clip_by_value(pos_indices, clip_value_min=0, clip_value_max=(self.seq_length-1)*2),tf.int32))
        # pos_bias = tf.einsum('bhqd,qkd->bhqk', q, pos_enc)
        pos_bias = tf.transpose(pos_enc, perm=(0, 3, 1, 2))
        logits = matmul_qk + pos_bias
        
        # Apply the mask to set attention scores for filled positions to a very large negative value
        # mask = tf.cast(valid_mask, dtype=tf.float32)
        # mask = tf.expand_dims(mask, axis=1)  # For head
        # mask = tf.expand_dims(mask, axis=2)  # For seq_length
        large_negative_value = -1e9
        logits = logits * valid_mask + (1.0 - valid_mask) * large_negative_value
        
        attention_weights = tf.nn.softmax(logits, axis=-1)  # (batch_size, num_heads, seq_length, seq_length)
        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_length, depth)
        
        return output
    # self = attention_layer
    # valid_mask=padding_maskL
    # q, v, k = xL, xL, xL 
    # position_array = x_doy
    def call(self, q, v, k, position_array, valid_mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_length, d_model)
        k = self.wk(k)  # (batch_size, seq_length, d_model)
        v = self.wv(v)  # (batch_size, seq_length, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_length, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_length, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_length, depth)
        
        scaled_attention = self.scaled_dot_product_attention(q, k, v, position_array, valid_mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_length, num_heads, depth)
        
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_length, d_model)
        
        output = self.dense(concat_attention)  # (batch_size, seq_length, d_model)
        
        return output


# Example usage
# batch_size = 2
# seq_length = 10
# d_model = 16
# num_heads = 4

# Create dummy queries, keys, values, and position array
# queries = tf.random.uniform((batch_size, seq_length, d_model))
# keys = tf.random.uniform((batch_size, seq_length, d_model))
# values = tf.random.uniform((batch_size, seq_length, d_model))
# position_array = tf.constant([[1, 2, 3, -9999, 5, 6, -9999, 8, 9, 10], [1, -9999, 3, 4, 5, 6, 7, 8, -9999, 10]], dtype=tf.int32)
# valid_mask = tf.not_equal(position_array, filled_value) # need to be changed 

# Instantiate the layer
# attention_layer = MultiHeadAttentionWithRelativePositionBias(d_model, num_heads, seq_length)

# Compute the output
# output = attention_layer(values, keys, queries, position_array, valid_mask)
# print(output)


# get this from ChatGPT
# 
def self_attention_with_relative_position_bias(queries, keys, values, position_array, relative_position_bias, filled_value=-9999):
    """
    Compute self-attention with relative position bias.
    
    Args:
    queries: Tensor of shape (batch_size, seq_length, d_model).
    keys: Tensor of shape (batch_size, seq_length, d_model).
    values: Tensor of shape (batch_size, seq_length, d_model).
    position_array: Tensor of shape (batch_size, seq_length) with positions or filled values.
    relative_position_bias: Tensor of shape (2*seq_length-1, d_model).
    filled_value: Scalar value indicating positions to be skipped.
    
    Returns:
    Tensor of shape (batch_size, seq_length, d_model).
    """
    batch_size, seq_length, d_model = tf.shape(queries)[0], tf.shape(queries)[1], tf.shape(queries)[2]
    # Create mask for non-filled positions
    valid_mask = tf.not_equal(position_array, filled_value)
    # Adjust position indices to be in the range of [0, 2*seq_length-2]
    A_expanded_i = tf.expand_dims(position_array, axis=2)  # shape [a, b, 1]
    A_expanded_j = tf.expand_dims(position_array, axis=1)  # shape [a, 1, b]  
    result = A_expanded_i - A_expanded_j + seq_length - 1 #     
    pos_indices = position_array + seq_length - 1
    # Mask out the invalid positions in pos_indices
    # pos_indices = tf.where(valid_mask, pos_indices, tf.zeros_like(pos_indices))
    pos_indices = tf.where(valid_mask, pos_indices, tf.zeros_like(pos_indices))
    
    # Gather relative positional encodings
    pos_enc = tf.gather(relative_position_bias, pos_indices)
    
    # Compute dot product attention scores
    scores = tf.matmul(queries, keys, transpose_b=True)
    
    # Add relative positional bias
    scores += tf.einsum('bij,bjk->bik', pos_enc, tf.transpose(pos_enc, perm=[0, 2, 1]))
    
    # Apply mask to scores
    mask = tf.cast(valid_mask, dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=1)
    scores = scores * mask + (1.0 - mask) * tf.float32.min
    
    # Apply softmax to get attention weights
    attention_weights = tf.nn.softmax(scores, axis=-1)
    
    # Compute weighted sum of values
    output = tf.matmul(attention_weights, values)
    
    return output


class MultiHeadAttentionWithRelativePositionBias_CHATGPT(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, seq_length):
        super(MultiHeadAttentionWithRelativePositionBias, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.seq_length = seq_length
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
        
        # Initialize learnable relative position bias
        self.relative_position_bias = self.add_weight(
            "relative_position_bias",
            shape=[2 * seq_length - 1, self.depth],
            initializer="random_normal",
            trainable=True
        )
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_length, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, pos_enc, position_array, filled_value):
        """Calculate the attention weights."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_length, seq_length)
        
        # Adjust position indices to be in the range of [0, 2*seq_length-2]
        pos_indices = position_array + self.seq_length - 1
        valid_mask = tf.not_equal(position_array, filled_value)
        pos_indices = tf.where(valid_mask, pos_indices, tf.zeros_like(pos_indices))
        
        # Gather relative positional encodings
        pos_enc = tf.gather(pos_enc, pos_indices)
        
        pos_bias = tf.einsum('bhqd,qkd->bhqk', q, pos_enc)
        
        logits = matmul_qk + pos_bias
        
        # Apply the mask to set attention scores for filled positions to a very large negative value
        mask = tf.cast(valid_mask, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=1)  # For head
        mask = tf.expand_dims(mask, axis=2)  # For seq_length
        large_negative_value = -1e9
        logits = logits * mask + (1.0 - mask) * large_negative_value
        
        attention_weights = tf.nn.softmax(logits, axis=-1)  # (batch_size, num_heads, seq_length, seq_length)
        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_length, depth)
        
        return output
    
    def call(self, v, k, q, position_array, filled_value=-9999):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_length, d_model)
        k = self.wk(k)  # (batch_size, seq_length, d_model)
        v = self.wv(v)  # (batch_size, seq_length, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_length, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_length, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_length, depth)
        
        scaled_attention = self.scaled_dot_product_attention(q, k, v, self.relative_position_bias, position_array, filled_value)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_length, num_heads, depth)
        
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_length, d_model)
        
        output = self.dense(concat_attention)  # (batch_size, seq_length, d_model)
        
        return output
