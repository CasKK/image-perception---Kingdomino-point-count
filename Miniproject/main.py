import cv2 as cv
import numpy as np
from collections import deque
import time
from multiprocessing import Pool, cpu_count
import ast #Til at hive data ind fra fil



Image_array = []
template_blue = cv.imread(r"bluetemplate.jpg",0)
template_red =  cv.imread(r"redtemplate.jpg",0)
# template_blueRocks = cv.imread(r"bluetemplateRocks.png",0)
# template_redRocks =  cv.imread(r"redtemplateRocks.png",0)
#threshold = 0.7
min_distance = 7
border = 10

def equalize_brightness(img):
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(ycrcb)
    hist = cv.calcHist([y],[0],None,[256],[0,256])
    hist_accumulative = np.cumsum(hist)
    total_pixels = hist_accumulative[255]
    clip_amount = total_pixels * 0.4 / 100.0
    min_gray = np.searchsorted(hist_accumulative, clip_amount)
    max_gray = np.searchsorted(hist_accumulative, total_pixels - clip_amount)
    if max_gray > min_gray:
        alpha = 255.0 / (max_gray - min_gray)
        beta = -min_gray * alpha
        y = cv.convertScaleAbs(y, alpha=alpha, beta=beta)
    ycrcb = cv.merge([y, cr, cb])
    result = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)
    # cv.imshow("Original", img)
    # cv.waitKey(0)
    # cv.destroyAllWindows() 
    # cv.imshow("Equalized Brightness", result)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return result

def meanBGR1(img):
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8)
    exclude_x1 = int(w * 0.25)
    exclude_x2 = int(w * 0.75)
    exclude_y1 = int(h * 0.25)
    exclude_y2 = int(h * 0.75)
    mask[exclude_y1:exclude_y2, exclude_x1:exclude_x2] = 0

    border = 5
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0

    masked_img = cv.bitwise_and(img, img, mask=mask)
    mean_bgr = cv.mean(img, mask=mask)
    # print(mean_bgr)
    # cv.imshow("Masked Image", masked_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return [mean_bgr[0], mean_bgr[1], mean_bgr[2], 0.0]

def meanBGR(img):
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8)
    h = h - border * 2
    w = w - border * 2
    exclude_x1 = int(w * 0.25)
    exclude_x2 = int(w * 0.75)
    exclude_y1 = int(h * 0.25)
    exclude_y2 = int(h * 0.75)
    mask[exclude_y1 + border:exclude_y2 + border, exclude_x1 + border:exclude_x2 + border] = 0
    
    border1 = 5 + border    
    mask[:border1, :] = 0
    mask[-border1:, :] = 0
    mask[:, :border1] = 0
    mask[:, -border1:] = 0

    masked_img = cv.bitwise_and(img, img, mask=mask)
    mean_bgr = cv.mean(img, mask=mask)
    # print(mean_bgr)
    # cv.imshow("Masked Image", masked_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return [mean_bgr[0], mean_bgr[1], mean_bgr[2], 0.0]

def knn(tile_img, training_data, imagenr, tilenr):
    k = 1
    edges = cv.Canny(tile_img[10:-10, 10:-10], 100, 255)
    #shadows = detect_shadow_blobs(img)
    edge_mean = float(np.mean(edges))
    mean_bgr = meanBGR(tile_img)
    afstande = []
    for trainingtile in training_data:
        afstand = np.sqrt((trainingtile[1][0] - mean_bgr[0])**2 + (trainingtile[1][1] - mean_bgr[1])**2 + (trainingtile[1][2] - mean_bgr[2])**2 + (trainingtile[1][3] - edge_mean)**2) # + (trainingtile[1][4]-shadows)**2
        afstande.append((afstand, trainingtile[0]))
    afstande1 = np.array(sorted(afstande, key=lambda x: x[0]))
    #print(afstande1[:k])

    unique_tiles, counts = np.unique(afstande1[:k, 1], return_counts=True)
    tile_type = str(unique_tiles[np.argmax(counts)])
    #print(unique_tiles)
    #print(counts)
    #print("Most common:", tile_type)
    return tile_type#, imagenr, tilenr, [mean_bgr[0], mean_bgr[1], mean_bgr[2], edge_mean]

def rotate_template(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotation matrix
    M = cv.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding dimensions
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust rotation matrix to account for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform rotation with expanded canvas
    return cv.warpAffine(image, M, (new_w, new_h))

def template_matching(img, template, threshold):
    res = cv.matchTemplate(img,template,cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    matches = []
    tile_with_rects = cv.cvtColor(img.copy(), cv.COLOR_GRAY2BGR)
    for pt in zip(*loc[::-1]):  # convert to (x, y)
        if all(np.linalg.norm(np.array(pt) - np.array(existing)) > min_distance for existing in matches):
            matches.append(pt)
        cv.rectangle(tile_with_rects, pt, (pt[0]+template.shape[1], pt[1]+template.shape[0]), (100,150,20), 2)

    #if len(matches) < 1:
        #print("No matches found")
    #matches = matches[:3]  # Keep only the first 3 matches
    #print(f"number_of_matches={len(matches)}")
    return matches, tile_with_rects, res

def split_into_tiles1(img):
    tiles = []
    h, w = img.shape[:2] 
    tile_h, tile_w = h // 5, w // 5
    for i in range(5):
        for j in range(5):
            tile = img[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            tiles.append(tile)
    return tiles

def split_into_tiles(img):
    tiles = []
    h, w = img.shape[:2] 
    #print(f"h={h}, w={w}")
    tile_h, tile_w = h // 5, w // 5

    img = cv.copyMakeBorder(img, border, border, border, border, cv.BORDER_CONSTANT, value=[0, 0, 0])
    
    for i in range(5):
        for j in range(5):
            tile = img[i*tile_h:(i+1)*tile_h+border*2, j*tile_w:(j+1)*tile_w+border*2]
            tiles.append(tile)
            # cv.imshow(f"Tile", tile)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
    return tiles

def count_crowns(img, biome, imagenr):
    total_crowns = 0
    if biome == "water":
        red_channel_img = cv.split(img)[2]
        template = template_red
        threshold = 0.8
        for i in range(4):
            template = rotate_template(template, 90)
            matches,matched_tile, res = template_matching(red_channel_img, template, threshold)
            #template_matching_diagnose(img, biome, imagenr, matches, matched_tile, res, total_crowns)
            if len(matches) >= 1:
                   break
    else:
        blue_channel_img = cv.split(img)[0]
        if biome == "rocks":
            threshold = 0.655
        # elif biome == "forest":
        #     threshold = 0.68
        # elif biome == "desert":
        #     threshold = 0.69
        else:
            threshold = 0.68
        template = template_blue
        for i in range(4):
            template = rotate_template(template, 90)
            matches,matched_tile, res = template_matching(blue_channel_img, template, threshold)
            total_crowns += len(matches)
            #template_matching_diagnose(img, biome, imagenr, matches, matched_tile, res, total_crowns)
            if len(matches) >= 1:
                   break
    #print(f"matches={total_crowns}")
    return len(matches), matched_tile 

def template_matching_diagnose(img, biome, imagenr, matches, matched_tile, res, total_crowns):
    if(imagenr == 12 or imagenr == 31): # len(matches) > 0
        print(f"Image {imagenr}, Biome: {biome}, Crowns: {total_crowns}")
        cv.imshow("img", matched_tile)
        cv.imshow("res", res)
        cv.waitKey(0)

def detect_shadow_blobs(image_path):
    # Load the image
    image = image_path
    if image is None:
        print("❌ Failed to load image.")
        return

    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Threshold to isolate dark regions (potential shadows)
    _, shadow_mask = cv.threshold(gray, 50, 155, cv.THRESH_BINARY_INV)

    # Morphological operations to clean up the mask
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    shadow_cleaned = cv.morphologyEx(shadow_mask, cv.MORPH_OPEN, kernel)

    # Find contours of the shadow blobs
    contours, _ = cv.findContours(shadow_cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # --- ✅ Compute shadow metrics ---
    shadow_pixels = cv.countNonZero(shadow_cleaned)
    total_pixels = gray.shape[0] * gray.shape[1]
    shadow_ratio = shadow_pixels / total_pixels  # between 0.0 and 1.0
    # ---------------------------------
    # Draw contours for visualization (optional)
    result_image = image.copy()
    cv.drawContours(result_image, contours, -1, (0, 255, 0), 2)

    #print(f"🕶️ Shadow area: {shadow_pixels} pixels ({shadow_ratio * 100:.2f}% of image)")

    return shadow_pixels

def calculate_score(tile_info):
    score = 0
    checked_tiles = np.zeros(len(tile_info), dtype=bool)

    rows, cols = 5, 5  # 5x5 grid

    for i in range(len(tile_info)):
        if checked_tiles[i]:
            continue  # skip already counted tiles

        queue = deque([i])
        checked_tiles[i] = True
        tile_type = tile_info[i]["type"]

        cluster_count = 0   # number of tiles in cluster
        cluster_crowns = 0  # sum of crowns in cluster

        while queue:
            idx = queue.popleft()
            cluster_count += 1
            cluster_crowns += tile_info[idx]["crowns"]

            r, c = divmod(idx, cols)

            neighbors = []
            if r > 0:  # up
                neighbors.append(idx - cols)
            if r < rows - 1:  # down
                neighbors.append(idx + cols)
            if c > 0:  # left
                neighbors.append(idx - 1)
            if c < cols - 1:  # right
                neighbors.append(idx + 1)

            for n in neighbors:
                if not checked_tiles[n] and tile_info[n]["type"] == tile_type:
                    queue.append(n)
                    checked_tiles[n] = True

        # multiply sum of crowns by number of tiles in cluster
        score += cluster_count * cluster_crowns

    print(f"score={score}")
    return score

time_start = time.time()

training_data = []
with open(fr"Data/TrainingDataAllTiles_Type_Img_BGR_Canny_Shadow.txt") as f:
    for i, line in enumerate(f):
        #if i >= 1175:
        #    break
        line = line.strip()
        if line:
            tile_type, _, _, values = ast.literal_eval(line)
            training_data.append((tile_type, values))


for i in range(74):
    filename = fr"King_Domino_dataset\Cropped_and_perspective_corrected_boards\{i+1}.jpg"
    img = cv.imread(filename)
    Image_array.append(img)
    #detect_shadow_blobs(img)
    #print(f"{i} image done")

with open("Data/AllImages.txt", "w") as f:
    f.write("")

for i in range(74):
    dt = np.dtype([("type", "U10"), ("crowns", int)])
    tile_data = np.zeros(25, dtype=dt)
    img_temp = equalize_brightness(Image_array[i])
    tiles = split_into_tiles(img_temp)
    for j in range(len(tiles)):
        tile_data[j]["type"] = knn(tiles[j], training_data, i, j)
        #print(f"Image {i+1}, Tile {j+1}, Type: {tile_data[j]['type']}")
        tile_data[j]["crowns"] = count_crowns(tiles[j], tile_data[j]["type"], i)[0]
        with open("Data/AllImages.txt", "a") as file:
            file.write(f"{tile_data[j]["type"]}, {i}, {j}, {tile_data[j]["crowns"]}\n")
    total_score = calculate_score(tile_data)
    with open("Data/AllImages.txt", "a") as file:
        file.write(f"Total points for image {i} = {total_score}\n")
    #print(f"Total score for image {i+1}: {total_score}")



def normalize1(line):
    return [x.strip() for x in line.split(',')]

def compare_files1():
    with open("Data/CorrectTileCrownPoint.txt", "r") as f:
        old_data = [normalize1(line) for line in f if line.strip()]
        
    with open("Data/AllImages.txt", "r") as f:
        new_data = [normalize1(line) for line in f if line.strip()]

    # Ensure same length
    max_len = max(len(new_data), len(old_data))
    old_data.extend([[""]] * (max_len - len(old_data)))
    new_data.extend([[""]] * (max_len - len(new_data)))

    # Compare and format output
    diff_lines = []
    for idx, (new_line, old_line) in enumerate(zip(new_data, old_data)):
        if new_line != old_line:
            new_str = ", ".join(new_line)
            old_str = ", ".join(old_line)
            diff_lines.append(f"{idx:04d} | {new_str:<35} | {old_str}")

    # Write results
    if diff_lines:
        with open(fr"Data\differences6.txt", "w") as f:
            for line in diff_lines:
                f.write(line + "\n")
        print("Found", len(diff_lines), "differences. Written to 'differences6.txt'.")
    else:
        print("No differences found!")

compare_files1()













time_end = time.time()  # end timer
print(f"Runtime: {time_end - time_start:.4f} seconds")










