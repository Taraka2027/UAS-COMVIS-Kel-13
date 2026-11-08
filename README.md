# UAS Computer Vision - Kelompok 13

Lanjutan UTS (Weizmann Horse Database, segmentasi objek): 6 kandidat enhancement/filter -> pilih terbaik (MSE/PSNR)
-> Canny dan Laplacian (metode klasik) -> U-Net dan DeepLabV3 (deep learning), total 8 output.

## Cara menjalankan
    pip install -r requirements.txt
    python uas_tahap1.py --data weizmann_horse_db     # enhancement, evaluasi 6 kandidat, Canny, Laplacian, ekspor data_dl.npz
    python uas_dl.py --epochs 60                      # U-Net dan DeepLabV3 (dilatih dari nol)

`uas_dl.py` menghitung peta tepi Canny/Laplacian sendiri dari citra asli, lalu melatih U-Net dan DeepLabV3 dengan
tiga jenis masukan untuk tiap masukan enhancement: citra enhancement langsung (pembanding), peta tepi Canny,
dan peta tepi Laplacian. Hasil ada di `hasil/` (dl_metrik.csv dan fig_dl_*.png).

Anggota: Gunawan Wibisono (187231041), Balya Ananta Wicaksana Annahid (187231033),
Abyan Arif Izzat Satrio (187231034), Muhammad Herjuna Taraka (187231042)
