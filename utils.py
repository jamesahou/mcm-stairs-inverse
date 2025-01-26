
in2pix_factor = 2

def in2pix(inches):
    return inches * in2pix_factor

def pix2in(pixels):
    return pixels // in2pix_factor