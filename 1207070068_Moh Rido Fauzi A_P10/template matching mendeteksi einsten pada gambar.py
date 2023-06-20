import cv2
# tampilkan kedua gambar
from matplotlib import pyplot as plt

# panggil dan konversi warna agar sesuai dengan Matplotlib
gambar = cv2.imread('rido2.png')
gambar = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB) # simpan dengan nama yang sama = ditumpuk

# panggil dan konversi warna agar sesuai dengan Matplotlib
img = cv2.imread('rido.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.subplot(121),plt.imshow(gambar), plt.title('rido')
plt.subplot(122),plt.imshow(img), plt.title('Foto Bersama')
plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('rido.jpg', 0)
img2 = img.copy()
template = cv2.imread('rido2.png', 0)
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# Perbesar ukuran hasil plotting
plt.rcParams["figure.figsize"] = (15, 6)

# Membuat subplot dengan jumlah metode
num_methods = len(methods)
fig, axes = plt.subplots(2, num_methods, sharex=True, sharey=True)

for i, met in enumerate(methods):
    img = img2.copy()
    method = eval(met)

    # Menggunakan template matching
    res = cv2.matchTemplate(img, template, method)

    # Mencari ukuran citra template untuk menggambar kotak
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Metode TM_SQDIFF dan TM_SQDIFF_NORMED menggunakan persamaan yang sedikit berbeda
    # sehingga dibuatkan fungsi khusus untuk mengambil nilai minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Buat persegi pada lokasi yang ditemukan
    cv2.rectangle(img, top_left, bottom_right, 255, 2)  # 2 adalah ketebalan garis kotak

    # Tampilkan hasil matching dan lokasi terdeteksi pada subplot yang sesuai
    axes[0, i].imshow(res, cmap='gray')
    axes[0, i].set_title('Hasil matching')
    axes[0, i].axis('off')

    axes[1, i].imshow(img, cmap='gray')
    axes[1, i].set_title('Lokasi terdeteksi')
    axes[1, i].axis('off')

# Atur tata letak subplot
plt.tight_layout()

# Tampilkan gambar
plt.show()
