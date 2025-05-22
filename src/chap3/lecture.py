import argparse, torch
from src.chap3.naive_activations import softmax_naive
from src.chap3.self_attention import SelfAttention
from src.chap3.causal_attention import CausalAttention
from src.chap3.naive_multihead_attention import MultiHeadAttentionWrapper
from src.chap3.multihead_attention import MultiHeadAttention

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=int, default=0)
    parser.add_argument("-s", "--section", type=int, default=0)

    args = parser.parse_args()
    inputs = torch.tensor([
        [0.43, 0.15, 0.89], # Your
        [0.55, 0.87, 0.66], # journey
        [0.57, 0.85, 0.64], # starts
        [0.22, 0.58, 0.33], # with
        [0.77, 0.25, 0.10], # one
        [0.05, 0.80, 0.55], # step
    ])

    # ----- 3.3.1 simple self-attention mechanism ----- #
    if args.mode == 3 and args.section == 1:
        print("# ----- 3.3.1 self-attention mechanism ----- #")

        query = inputs[1] # journey as query
        attn_scores_2 = torch.empty(inputs.shape[0])
        for i, x_i in enumerate(inputs):
            attn_scores_2[i] = torch.dot(x_i, query)
        print("the result of self-attention on the second word 'journey':\n", attn_scores_2)

        attn_weights_score_2 = attn_scores_2 / attn_scores_2.sum()
        print("attention weights:\n", attn_weights_score_2)
        print("sum: ", attn_weights_score_2.sum())

        attn_weights_2_naive = softmax_naive(attn_scores_2)
        print("attention weighted by naive softmax:\n", attn_weights_2_naive)
        print("sum: ", attn_weights_2_naive.sum())

        attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
        print("attention weighted by torch softmax:\n", attn_weights_2)
        print("sum: ", attn_weights_2)
        print("")

    # ----- 3.3.2 simple attention weights for all input tokens ----- #
    if args.mode == 3 and args.section == 2:
        print("# ----- 3.3.2 attention weights for all input tokens ----- #")

        '''
        slow version of self attention
        num_tok = inputs.shape[0]
        attn_scores = torch.empty(num_tok, num_tok)
        for i, x_i in enumerate(inputs):
            for j, x_j in enumerate(inputs):
                attn_scores[i][j] = torch.dot(x_i, x_j)
        '''
        attn_scores = inputs @ inputs.T
        print("attention score across all words:\n", attn_scores)
        attn_weights = torch.softmax(attn_scores, dim=1)
        print("attention weighted by softmax:\n", attn_weights)
        all_context_vecs = attn_weights @ inputs
        print("result context vectors:\n", all_context_vecs)
        print("")

    # ----- 3.4 self-attention with trainable weights ----- #
    if args.mode >= 4:
        x_2 = inputs[1]
        d_in = inputs.shape[1]
        d_out = 2

    # ----- 3.4.1 computing attention weights step by step ----- #    
    if args.mode == 4 and args.section == 1:
        print("# ----- 3.4.1 computing attention weights step by step ----- #")
        
        torch.manual_seed(123)
        w_q = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        w_k = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        w_v = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

        q_2 = x_2 @ w_q
        k_2 = x_2 @ w_k
        v_2 = x_2 @ w_v
        print("the queries weighted by w_q:\n", q_2)

        keys = inputs @ w_k
        values = inputs @ w_v
        print("the shape of weighted keys:\n", keys.shape)

        attn_score_22 = q_2.dot(k_2)
        print("attention score of word 2 querying itself:\n", attn_score_22)
        attn_scores_2 = q_2 @ keys.T
        print("attention scores of word 2 on all weighed input keys:\n", attn_scores_2)

        d_k = keys.shape[-1]
        attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
        print("attention weights:\n", attn_weights_2)

        context_vec_2 = attn_weights_2 @ values
        print("attention context vector:\n", context_vec_2)

    # ----- 3.4.1 computing attention weights step by step ----- #   
    if args.mode == 4 and args.section == 2:
        torch.manual_seed(789)
        sa = SelfAttention(d_in, d_out)
        print("outputted result from naive self attention:\n", sa(inputs))
    
    # ----- 3.5 causal attention / masked attention ----- #
    if args.mode == 5 and args.section <= 2:
        torch.manual_seed(789)
        sa = SelfAttention(d_in, d_out)
        queries = sa.w_q(inputs)
        keys = sa.w_k(inputs)
        attn_scores = queries @ keys.T
        context_len = attn_scores.shape[0]

    # ----- 3.5.1 applying a causal attention mask ----- #  
    if args.mode == 5 and args.section == 1:
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        print("the attention weight matrix without causal masking:\n", attn_weights)

        mask_simple = torch.tril(torch.ones(context_len, context_len))
        # print("the causal mask implemented:\n", mask_simple)
        masked_simple = attn_weights * mask_simple
        print("masked causal attention weights:\n", masked_simple)

        # renormalization
        row_sums = masked_simple.sum(dim=1, keepdim=True)
        masked_simple_norm = masked_simple / row_sums
        print("re-normalized masked causal mask:\n", masked_simple_norm)

        # instead utilizing softmax to do the trick
        mask = torch.triu(torch.ones(context_len, context_len), 
                          diagonal=1)
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        print("alternative masking using softmax, masking null values to -inf:\n", masked)
        attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
        print("attention weight after this masking scheme:\n", attn_weights)

    # ----- 3.5.2 applying a dropout layer ----- #  
    if args.mode == 5 and args.section == 2:
        print("# ----- 3.5.2 applying a dropout layer ----- #")
        mask = torch.triu(torch.ones(context_len, context_len), 
                          diagonal=1)
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
        print("attention weights before dropout:\n", attn_weights)
        torch.manual_seed(123)
        dropout_layer = torch.nn.Dropout(0.5)
        print("attention weights after dropout of 0.5:\n", dropout_layer(attn_weights))

    # ----- 3.5.3 implementing a compact causal attention class ----- #  
    if args.mode == 5 and args.section == 3:
        print("# ----- 3.5.3 implementing a compact causal attention class ----- #")
        batch = torch.stack((inputs, inputs), dim=0)
        print("shape of a two-item batch input:\n", batch.shape)

        torch.manual_seed(123)
        context_len = batch.shape[1]
        ca = CausalAttention(d_in, d_out, context_len, 0.0)
        context_vecs = ca(batch)
        print("contxt vecs shape:\n", context_vecs.shape)
    
    # ----- 3.6 extending to multi-head attention ----- #
    if args.mode == 6:
        print("# ----- 3.6 extending to multi-head attention ----- #")
        torch.manual_seed(123)
        batch = torch.stack((inputs, inputs), dim=0)

    # ----- 3.6.1 naive implementation: stacking multiple single-head attention layers ----- #
    if args.mode == 6 and args.section == 1:
        print("# ----- 3.6.1 naive implementation: stacking multiple single-head attention layers ----- #")
        context_len = batch.shape[1]
        mha = MultiHeadAttentionWrapper(d_in, d_out, context_len, 0.0, 2)
        context_vecs = mha(batch)

        print("shape of the context vector:\n", context_vecs.shape)
        print("the context vector from concatenating multiple heads:\n", context_vecs)
    
    # ----- 3.6.2 multi-head attention with weight splits ----- #
    if args.mode == 6 and args.section == 2:
        print("# ----- 3.6.2 multi-head attention with weight splits ----- #")
        batch_size, context_len, d_in = batch.shape
        mha = MultiHeadAttention(d_in, d_out, context_len, 0.0, num_heads=2)
        context_vecs = mha(batch)
        print("shape of the context vector:\n", context_vecs.shape)
        print("the context vector from splitting multiple heads:\n", context_vecs)



    
