# output of neuron = Y = f(w1.X1 + w2.x2 +b)
# 23, 75, 176
#h1  X: -0.46122 Y: 0.97314 Z: -0.39203 -> -0.81546
#h2 X: 0.78548 Y: 2.10584 Z: -0.57847 -> 1.03375

# out
import math

def forwardPass(wiek, waga, wzrost):
    hidden1 = -0.46122 * wiek + 0.97314 * waga + -0.39203 * wzrost + 0.80109
    hidden1_po_aktywacji = 1 / (1 + math.exp(-hidden1))
    hidden2 = 0.78548 * wiek + 2.10584 * waga + -0.57847 * wzrost + 0.43529
    hidden2_po_aktywacji = 1 / (1 + math.exp(-hidden2))
    output = hidden1_po_aktywacji * -0.81546  + hidden2_po_aktywacji * 1.03775 + -0.2368
    return output

def run():
    wiek = 23
    waga = 75
    wzrost = 176

    result = forwardPass(wiek, waga, wzrost)
    print(result)
