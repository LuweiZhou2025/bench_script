
thread layout:
    thr_layout = fx.make_layout((4, 16), (16, 1))
    thr_layout = fx.make_layout((16, 4), (1, 16))
这里的thread(4, 16)代表m维度4个thread, n维度1个thread?

preshuffle_layout_B = fx.make_layout(((16, N // 16), (8, 4, K // 32)), ((8, 16 * K), (1, 8 * 16, 8 * 16 * 4)))


make_tiled_copy needs: copy_atom, thread-value layout, tiled_mn. 整个的tiled copy也可以是一个sparse的tile。
copy atom ：每个thread每次搬运的最少数据的数目决定用哪个指令集。
tv layout：每个thread可以根据自己的thread id以及value id 算出的index. 每个thread copy的每个value, 都有对应的一维index. 
            一个thread里面的index在物理的存储空间上可以连续也可以不连续。
            tv 值对应的index是一维。这个index使用column-major(N major)的方式计算的。 对于矩阵[M, N]里的[i,j], index = j * M  + i.
            注意这里的index ->[i, j] 对应的第i行，第j列，并不是物理存储意义上的第i行和第j列，而只是这个逻辑上的第i行第j列。
            具体的将逻辑上的[i,j]对应到真实layout对立面的数值是，需要注意flexDSL里面的mode的有序的，每个mode里面的dimension也是有序的。
            所以对于一种shape reorder/preshuffle之后只有一种表达，比如说Matrix B bf16 [N,K]按照aiter里面的preshuffle之后layout
            [N//16, K//32, 4k, 16n, 8k]，使用flexDSL表达的时候只有一种
            fx.make_layout(((16, N // 16), (8, 4, K // 32)), ((8, 16 * K), (1, 8 * 16, 8 * 16 * 4)))
            N 在mode0, K在mode1, 里面的sublayout也是有序的，变化最快的在sublayout的mode0,
          
tiled_mn:  每个元素的一维index对应的矩阵里面逻辑上的行与列。 

        
            
