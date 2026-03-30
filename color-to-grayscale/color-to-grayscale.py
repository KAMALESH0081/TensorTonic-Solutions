def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    result = []
    
    len_i = len(image)
    len_j = len(image[0])
    
    for i in range(len_i):
        row = []
        for j in range(len_j):
            y = image[i][j][0] * 0.299 + image[i][j][1] * 0.587 + image[i][j][2] * 0.114
            row.append(y)

        result.append(row)

    return result
            
            