import cv2
import numpy as np
import joblib
import chess
from PIL import Image
import chess.engine


white = joblib.load('white.pkl')
black = joblib.load('black.pkl')
sc = joblib.load('scaler.pkl')
threshold_value = 128

m= 1
height= 800
width= 800
threshold_value = 128

board_array = [['' for _ in range(8)] for _ in range(8)]

def process_chess_image(image_file, turn):
    global m, height, width, threshold_value, board_array
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    def mse(imageA, imageB):
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err

    lower_green = np.array([60, 60, 60]) 
    upper_green = np.array([255, 255, 255]) 
    mask = cv2.inRange(image, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    cropped_board = image[y:y+h, x:x+w]
    cv2.imwrite('cropped.png', cropped_board)
    img1c = cv2.imread('cropped.png')  
    gray = cv2.cvtColor(img1c, cv2.COLOR_BGR2GRAY)
    board = cv2.resize(gray, (800, 800))
    board = Image.fromarray(board)
    board.save('board_test.jpg')
    board= Image.open('board_test.jpg')

    for n in range(1, 9):
        height1 = n*100
        h1= (n-1)*100
        for s in range (1, 9):
            width1 = s*100
            w1= (s-1)*100
            pic= board.crop((h1,w1, height1, width1)) 
            pic.save('dummy.jpg')
            pic = cv2.imread("dummy.jpg") 
            lower = np.array([0,0,0])
            upper = np.array([90,90,90])
            lower1 = np.array([248,248,248])
            upper1 = np.array([256,256,256])
            mask1 = cv2.inRange(pic, lower, upper)
            mask2 = cv2.inRange(pic, lower1, upper1)
            cmask = cv2.bitwise_or(mask1, mask2)
            output = cv2.bitwise_and(pic,pic, mask= cmask)
            kernel = np.ones((5,5),np.float32)/25
            dst = cv2.filter2D(output,-1,kernel)
            resized_image = cv2.resize(dst, (20, 20), interpolation=cv2.INTER_AREA)
            imageA = resized_image
            imageBG= cv2.imread(f'bg.jpg')
            if np.any(imageA > threshold_value):
                flattened_image = imageA.flatten()
                row2 = flattened_image.tolist()
                row2 = np.array(row2).reshape(1, -1)
                scaled_features = sc.transform(row2)
                prediction = white.predict(scaled_features)
                board_array[s-1][n-1]=prediction[0]
            elif mse(imageA, imageBG) < 50:
                board_array[s-1][n-1]=''
            else:
                flattened_image = imageA.flatten()
                row1 = flattened_image.tolist()
                row1 = np.array(row1).reshape(1, -1)
                scaled_features = sc.transform(row1)
                prediction = black.predict(scaled_features)
                board_array[s-1][n-1]=prediction[0]
            m+=1

    board = chess.Board.empty()

    for i in range(8):
        for j in range(8):
            piece = board_array[i][j]
            if piece:
                board.set_piece_at(chess.square(j, 7 - i), chess.Piece.from_symbol(piece))

    board.turn = chess.WHITE if turn == 'white' else chess.BLACK

    board = chess.Board(board.fen())

    engine_path = "Enter Your Chess Engine Path"

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    result = engine.play(board, chess.engine.Limit(time=1.0))
    fen = result.move.uci()
    engine.quit()
    return fen