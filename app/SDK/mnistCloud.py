from socket import *
import base64


def mnistRec(filename, threshold=0.8, xStride=28, yStride=28):
    imageFile = open(filename, "rb");
    image = base64.b64encode(imageFile.read());
    pktSend = "xs request_mnist_recognition " + image.decode("utf-8") + " " + str(threshold) + " " + str(xStride) + " " + str(yStride) + " *";
    addr = "127.0.0.1";
    sock = socket(AF_INET, SOCK_STREAM);
    sock.connect((addr, 777));
    sock.send(pktSend.encode(encoding="utf-8"));
    bytesGet = sock.recv(65536);  # too many will be cut
    result = bytesGet.decode("utf-8").split(" ");

    return result[2: ];