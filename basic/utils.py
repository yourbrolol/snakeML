def pool2d(input, kernel, stride):
    return [input[y:kernel[1]+y, x:kernel[0]+x] for y in range(0, input.shape[1]-1, stride[1]) for x in range(0, input.shape[0]-1, stride[0])]
