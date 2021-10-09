import numpy
import scipy.special
from tkinter import *
import tkinter as tk
from PIL import ImageGrab, Image
import imageio


# распознавание нарисованной цифры
def predictDigit():
    Image.open("digit.png").resize((28, 28), Image.LANCZOS).save("digit.png")
    imgArray = imageio.imread("digit.png", as_gray=True)
    imgData = 255.0 - imgArray.reshape(784)
    imgData = (imgData / 255.0 * 0.99) + 0.01

    outputs = n.query(imgData)
    digit = numpy.argmax(outputs)

    return digit


# класс нейронной сети
class Network:

    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes
        self.lr = learningRate

        # Матрицы весовых коэффициентов связей
        # wih (между входным и скрытым слоями) и
        # who (между скрытым и выходным слоями)
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # Функция активации
        self.activationFunction = lambda x: scipy.special.expit(x)

    # тренировка
    def train(self, inputsList, targetsList):
        # преобразование входных значений в двухмерный массив
        inputs = numpy.array(inputsList, ndmin=2).T
        targets = numpy.array(targetsList, ndmin=2).T

        # рассчитать входящие сигналы для скрытого слоя
        hiddenInputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hiddenOutputs = self.activationFunction(hiddenInputs)

        # рассчитать входящие сигналы для выходного слоя
        finalInputs = numpy.dot(self.who, hiddenOutputs)
        # рассчитать исходящие сигналы для выходного слоя
        finalOutputs = self.activationFunction(finalInputs)

        # ошибки выходного слоя
        outputErrors = targets - finalOutputs
        # ошибки скрытого слоя
        hiddenErrors = numpy.dot(self.who.T, outputErrors)

        # обновление веса для связей между скрытым и выходным слоями
        self.who += self.lr * numpy.dot((outputErrors * finalOutputs * (1.0 - finalOutputs)),
                                        numpy.transpose(hiddenOutputs))

        # обновление весовые коэффициенты для связей между входным и скрытым слоями
        self.wih += self.lr * numpy.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)),
                                        numpy.transpose(inputs))

    # опрос нейронной сети
    def query(self, inputsList):
        # преобразование входных значений в двухмерный массив
        inputs = numpy.array(inputsList, ndmin=2).T

        # рассчитать входящие сигналы для скрытого слоя
        hiddenInputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hiddenOutputs = self.activationFunction(hiddenInputs)

        # рассчитать входящие сигналы для выходного слоя
        finalInputs = numpy.dot(self.who, hiddenOutputs)
        # рассчитать исходящие сигналы для выходного слоя
        finalOutputs = self.activationFunction(finalInputs)

        return finalOutputs


# окно рисования
class Window(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        self.canvas = tk.Canvas(self, width=500, height=500, bg="white", cursor="cross")
        self.label = tk.Label(self, text=" ", font=("Helvetica", 14))
        self.buttonClear = tk.Button(self, text="Очистить", command=self.clearAll)
        self.classifyBtn = tk.Button(self, text="Распознать", command=self.classifyHandwritten)

        self.canvas.grid(row=0, column=0, columnspan=2)
        self.label.grid(row=1, column=0, columnspan=2, pady=10)
        self.buttonClear.grid(row=2, column=0, pady=10)
        self.classifyBtn.grid(row=2, column=1, pady=10)

        self.canvas.bind("<B1-Motion>", self.drawLines)

    # очистка холста
    def clearAll(self):
        self.canvas.delete("all")
        self.label.configure(text=" ")

    # вызов распознавания
    def classifyHandwritten(self):
        img = ImageGrab.grab((25, 50, 529, 554))
        img.save("digit.png")
        digit = predictDigit()

        self.label.configure(text="Распознанная цифра: " + str(digit))

    # рисование
    def drawLines(self, event):
        self.x = event.x
        self.y = event.y
        r = 15
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')


# количество входных, скрытых и выходных нейронов
inputNodes = 784
hiddenNodes = 256
outputNodes = 10

# коэффициент скорости обучения
learningRate = 0.1

n = Network(inputNodes, hiddenNodes, outputNodes, learningRate)

# загрузка тренировочного набора данных
trainingDataFile = open("mnist_train.csv", 'r')
trainingDataList = trainingDataFile.readlines()
trainingDataFile.close()

# тренировка нейронной сети
epochs = 5

for i in range(epochs):
    # перебрать все записи в тренировочном наборе данных
    for record in trainingDataList:
        allValues = record.split(',')
        # масштабировать и сместить входные значения
        inputs = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(outputNodes) + 0.01
        targets[int(allValues[0])] = 0.99
        n.train(inputs, targets)

# загрузка тестового набора данных
testDataFile = open("mnist_test.csv", 'r')
testDataList = testDataFile.readlines()
testDataFile.close()

# тестирование нейронной сети
score = []
for record in testDataList:
    allValues = record.split(',')
    # правильный ответ - первое значение
    correctLabel = int(allValues[0])
    # масштабировать и сместить входные значения
    inputs = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
    # опрос сети
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    # присоединить оценку ответа сети к концу списка
    if label == correctLabel:
        score.append(1)
    else:
        score.append(0)

scoreArray = numpy.asarray(score)
print("Точность:", scoreArray.sum() / scoreArray.size)

# вызов окна рисования
window = Window()
window.geometry('%dx%d+%d+%d' % (520, 600, 0, 0))
mainloop()
