import numpy as np
from keras.models import Model, load_model
from socket import *
import base64
import random
import matplotlib.pyplot as plt



class mnist:
    receiveSocket = socket(AF_INET, SOCK_STREAM);
    newSocket = socket();


    def __init__(self):
        self.model = load_model("model.h5");


    def predict(self, X, threshold):
        # make predictions
        Y = self.model.predict(X);
        Y_pred = np.argmax(Y, axis=1);
        Y_prob = np.max(Y, axis=1);
        ans = [];
        for i in range(len(Y_prob)):
            if Y_prob[i] >= threshold and Y_pred[i] != 1:
                ans.append((i, Y_pred[i]));

        return ans;


    def runServer(self):
        bufsize = self.newSocket.getsockopt(SOL_SOCKET, SO_SNDBUF);
        print("Buffer size [Before]:" + str(bufsize));
        self.receiveSocket.bind(('', 777));
        self.receiveSocket.listen(5);
        print("Server is running on port 777!");
        while 1:
            self.newSocket, addr = self.receiveSocket.accept();
            tot = "";
            count = 0;
            while 1:
                count += 1;
                sentence = self.newSocket.recv(65535);
                #status = 0;
                tempStr = sentence.decode("utf-8");
                tot += tempStr;
                tempResults = tempStr.split(" ");
                print(tempResults);
                if tempResults[len(tempResults) - 1] == "*" or count >= 128:
                    break;

            results = tot.split(" ");
            print(results);
            if len(results) != 7:
                sendPkt = "xs answer_mnist_recognition -3 need_exactly_6_fields".encode(encoding="utf-8");
                print(sendPkt);
                self.newSocket.send(sendPkt);
                self.newSocket.close();
                continue;
            elif results[0] != "xs" or results[1] != "request_mnist_recognition":
                sendPkt = "xs answer_mnist_recognition -3 wrong_request_format".encode(encoding="utf-8");
                print(sendPkt);
                self.newSocket.send(sendPkt);
                self.newSocket.close();
                continue;

            elif len(results[2])%4 != 0 and "=" not in results[2]:
                sendPkt = "xs answer_mnist_recognition -3 base64_error".encode(encoding="utf-8");
                print(sendPkt);
                self.newSocket.send(sendPkt);
                self.newSocket.close();
                continue;

            elif int(results[4]) < 14 or int(results[5]) < 14 or float(results[3]) < 0.3:
                sendPkt = "xs answer_mnist_recognition -2".encode(encoding="utf-8");
                print(sendPkt);
                self.newSocket.send(sendPkt);
                self.newSocket.close();
                continue;

            else:
                self.onReceive(results[2], float(results[3]), int(results[4]), int(results[5]));
                self.newSocket.close();
                continue;


    def convert(self, rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b;
        return gray;


    def onReceive(self, code, threshold, xStride, yStride):
        # handle receive
        print((threshold, xStride, yStride));
        errPkt = "xs answer_mnist_recognition -1".encode(encoding="utf-8");
        image = base64.b64decode(code);
        tname = str(random.randint(1, 1000)) + ".jpg";
        fout = open(tname, "wb");
        fout.write(image);
        try:
            jpeg = plt._imread(tname);
        except OSError:
            print(errPkt);
            self.newSocket.send(errPkt);
            return;

        gray = self.convert(jpeg);
        if (gray.shape[0] * gray.shape[1] > 2000000 or gray.shape[0] < 28 or gray.shape[1] < 28):
            self.newSocket.send(errPkt);
            return;
        # sliding window
        m = gray.shape[0] - 28;
        m = m - m % yStride;
        n = gray.shape[1] - 28;
        n = n - n % xStride;
        X = np.zeros((int((m / yStride + 1) * (n / xStride + 1)), 784, 1));
        i = 0;
        j = 0;
        while 1:
            if i > m:
                break;
            X[int((i / yStride) * (n / xStride) + j / xStride), :, :] = gray[i: i + 28, j: j + 28].reshape(784, 1);
            j = j + xStride;
            if j > n:
                j = 0;
                i = i + yStride;

        print(X.shape);
        ans = self.predict(X / 255.0, threshold);
        res = "";  # return answer
        for obj in ans:
            pos, num = obj;
            y, x = divmod(pos, n / xStride);
            y = y * xStride;
            x = x * yStride;
            temp = "(" + str(int(x)) + "," + str(int(y)) + "," + str(num) + ") ";
            res = res + temp;

        sendPkt = ("xs answer_mnist_recognition " + str(len(ans)) + " " + res).encode(encoding="utf-8");
        print(sendPkt);
        self.newSocket.send(sendPkt);


