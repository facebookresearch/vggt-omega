1. Use multi layer or just last layer
2. Use one layer from DINO or not
3. Pixel shuffle at 1/4 or 1/16
4. Dimension number 256， 512？
5. 用concat 还是 add
6. 要不要把它分成
7. 要不要norm? 分开还是共享？
8. Interpolate, 还是Reassemble block?
9. 后处理的时候，DWConv 还是conv
10. proj_zero_init?
11. 要不要再加一个position embedding？










Ablations:
1. 直接原来的head，fitting 看看
2. 用DPT的方式，upsample 到1/4
3. 用interpolate + conv的方式，upsample 到1/4, DW conv
4. 用interpolate + conv的方式，upsample 到1/4, conv
5. 用interpolate， upsample 到1/4
6. 用interpolate + conv， upsample 到1/4， conv，但是仅用最后一层
7. 只用最后一level，但是加一层post conv
8. 




