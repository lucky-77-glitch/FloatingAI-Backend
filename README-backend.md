# Floating AI Backend

Proxy kecil yang menyimpan API key AI di server, sehingga key tidak pernah ada di APK.
Mendukung dua penyedia: **Gemini** (default, punya paket gratis) dan **Claude** (berbayar).

## Environment variable

| Nama | Wajib | Keterangan |
|------|-------|------------|
| `APP_TOKEN` | ya | Kunci akses buatan Anda sendiri (minimal 32 karakter acak). Dimasukkan juga di Pengaturan aplikasi |
| `AI_PROVIDER` | tidak | `gemini` (default) atau `anthropic` |
| `GEMINI_API_KEY` | jika provider gemini | Dari Google AI Studio (Get API key) |
| `ANTHROPIC_API_KEY` | jika provider anthropic | Dari platform.claude.com (Settings, API keys) |
| `AI_MODEL` | tidak | Default `gemini-2.5-flash-lite` (gemini) atau `claude-sonnet-5-5` (anthropic). Cek nama model yang tersedia di akun Anda |
| `AI_MAX_TOKENS` | tidak | Default 2048 (gemini) atau 1024 (anthropic) |

## Endpoint

- `GET /health` untuk uji koneksi (tanpa kunci).
- `POST /api/ask` dengan header `Authorization: Bearer <APP_TOKEN>`.

## Deploy ke Render (dari browser HP)

1. Buat repository GitHub baru khusus backend, lalu unggah `package.json` dan `server.js` ke root repo.
2. Di render.com, pilih **New, Web Service**, lalu hubungkan repo tadi.
3. Isi: Build Command `npm install`, Start Command `npm start`, Instance Type **Free**.
4. Di bagian Environment, tambahkan `GEMINI_API_KEY` dan `APP_TOKEN`.
5. Klik **Create Web Service** dan tunggu deploy selesai.
6. Buka `https://NAMA-SERVICE.onrender.com/health` di browser. Harus tampil `{"status":"ok"}`.

Instance gratis Render tidur setelah 15 menit tanpa permintaan, jadi permintaan pertama setelah itu bisa lambat.

## Catatan paket gratis

- Paket gratis Gemini punya batas permintaan per menit dan per hari, dan batas itu bisa berubah. Lihat halaman rate limits di ai.google.dev.
- Pada paket gratis, Google dapat memakai input dan output untuk meningkatkan produknya. Jangan kirim data pribadi atau rahasia.

## Keamanan

- Jangan pernah menaruh API key atau `APP_TOKEN` di repo.
- Server membatasi 20 permintaan per menit per IP dan ukuran body 64 KB.
- Isi percakapan tidak dicatat di log.
