def im2col(input, kernel, stride):
    Cin, H, W = input.shape
    KH, KW = kernel
    SH, SW = stride
    xr = range(0, W-1, SW)
    return [input[:, y:KW+y, x:KH+x] for y in range(0, H-1, SH) for x in xr]
