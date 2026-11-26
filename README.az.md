# Mətn Əsaslı Reel Redaktoru

Vertikal (9:16) Instagram Reels videolarını **transkriptini redaktə etməklə** montaj edin.
Klipi yükləyin, istəmədiyiniz sözlərin üstünü xətləyin, AI-ın təklif etdiyi B-roll
keçidlərini seçin və hazır Reel-i yandırılmış kinetik altyazılarla ixrac edin.

```
yükləmə ─► ffmpeg 16 kHz mono ─► Silero VAD ─► Groq whisper-large-v3 (söz-səviyyəli vaxt kodları)
        ─► LLM terminologiya düzəlişi ─► LLM B-roll anları ─► Pexels 9:16 axtarışı
        ─► [ mətni siz redaktə edirsiniz ] ─► zaman xəttinin yenidən hesablanması ─► ffmpeg render ─► MP4
```

- **Backend** — FastAPI (Python 3.12), tam tipləşdirilmiş, modulyar.
- **Frontend** — React 18 + TypeScript + Vite; interaktiv söz tokenləri, canlı
kəsim önizləməsi, sətir arasında B-roll yerləşdiriciləri.
- **Mock yoxdur.** Hər mərhələ real API-ni və ya real binary-ni çağırır.

---

## 1. Tələblər


| Komponent        | Versiya         | Qeyd                                                                                                    |
| ---------------- | --------------- | ------------------------------------------------------------------------------------------------------- |
| Python           | **3.10 – 3.13** | 3.12 tövsiyə olunur. PyTorch (Silero VAD) üçün hələ 3.14 wheel-ləri yoxdur.                             |
| FFmpeg + ffprobe | 6.0+            | `brew install ffmpeg` / `apt install ffmpeg`. **libass şiddətlə tövsiyə olunur** (bax §6).              |
| Node.js          | 18+             | Yalnız UI-nin build/servis edilməsi üçün.                                                               |
| Groq API açarı   | —               | [https://console.groq.com/keys](https://console.groq.com/keys)                                          |
| Pexels API açarı | —               | [https://www.pexels.com/api/](https://www.pexels.com/api/) (məcburi deyil; olmadıqda B-roll söndürülür) |


## 2. Quraşdırma

```bash
git clone <bu repo> && cd "AI Video editor"

# ---- backend -------------------------------------------------------------
python3.12 -m venv .venv
source .venv/bin/activate

# Tövsiyə olunan: PyTorch + Silero VAD daxildir
pip install -r requirements-vad.txt
# ...və ya yalnız əsas paketlər (daxili enerji əsaslı VAD-a keçir):
# pip install -r requirements.txt

# ---- açarlar -------------------------------------------------------------
cp .env.example .env
$EDITOR .env          # GROQ_API_KEY və PEXELS_API_KEY-i yazın

# ---- frontend ------------------------------------------------------------
cd frontend && npm install && npm run build && cd ..
```

### İşə salmaq

```bash
# tək proses: API + build edilmiş UI http://127.0.0.1:8000 ünvanında
./run.sh
```

Yaxud development zamanı iki hissəni ayrı işlədin:

```bash
uvicorn app.main:app --reload --port 8000     # terminal 1
cd frontend && npm run dev                    # terminal 2 -> http://localhost:5173
```

Vite dev serveri `/api` sorğularını `127.0.0.1:8000`-ə proxy edir, buna görə brauzer
tək origin-də qalır (`VITE_API_TARGET` ilə dəyişdirilə bilər).

Yoxlama nöqtələri: [`/api/health`](http://127.0.0.1:8000/api/health),
[`/api/capabilities`](http://127.0.0.1:8000/api/capabilities),
OpenAPI sənədləri [`/docs`](http://127.0.0.1:8000/docs).

## 3. İstifadə

1. **Yükləyin** — səs treki olan vertikal MP4/MOV. Emal fonda gedir; UI hər
 mərhələni göstərir.
2. **Mətni redaktə edin.** Sözün üstünü xətləmək üçün üzərinə klikləyin,
 diapazon üçün `Shift`+klik, bütün cümləni silmək üçün cümlənin üzərinə
 gəlib `✕`, geri qaytarmaq üçün `⌘/Ctrl+Z`. `Alt`+klik önizləməni o sözə aparır.
3. **Fasilələri sıxlaşdırın** (defolt olaraq aktivdir) — Silero VAD-ın sağ
 qalmış hissələrin *içində* tapdığı ölü havanı da kəsir; yalnız
 `SILENCE_TRIM_MS`-dən uzun sükutlar, deməli təbii ritm qorunur.
4. **B-roll seçin.** Hər təklifdə `Accept` / `Swap` (yeni Pexels sorğusu yazın) /
 `Dismiss` var. Açıq şəkildə qəbul edilmədən heç nə videoya yazılmır.
5. **Önizləmə** — pleyer silinmiş hissələri keçir, yəni render etməmişdən əvvəl
 real kəsimi eşidirsiniz.
6. **Render** edin və MP4-ü endirin.

## 4. Arxitektura

```
app/
  config.py            bütün parametrlər, .env-dən oxunur, bir dəfə validasiya edilir
  errors.py            tipləşdirilmiş exception-lar -> HTTP status + sabit xəta kodları
  schemas.py           Pydantic modelləri (həm də diskdəki format)
  ffmpeg_utils.py      probe, progress bildirən runner, imkanların yoxlanması, şrift tapılması
  audio_processing.py  16 kHz çıxarma, Silero VAD, sıxılmış ASR parçaları + vaxt xəritəsi
  transcription.py     Groq whisper-large-v3, cümlələrə bölmə, terminologiya düzəlişi
  llm_broll.py         LLM ilə an seçimi, Pexels 9:16 axtarışı, klip endirmə
  timeline_manager.py  mətn redaktəsi -> kəsim siyahısı -> yenidən vaxtlanmış sözlər və B-roll
  subtitles.py         kinetik altyazılar: .ass karaoke + Pillow raster mühərriki
  renderer.py          tək keçidli ffmpeg filtergraph (kəsim, birləşdirmə, örtük, yandırma)
  store.py             atomik layihə-başına JSON saxlama, layihə-başına kilidlər
  jobs.py              thread-pool tapşırıq icraçısı, progress diskə yazılır
  pipeline.py          ingest və render mərhələlərinin orkestrasiyası
  main.py              FastAPI marşrutları
  ui.py                build edilmiş SPA-nı servis edir
frontend/src/          React UI (components/, api.ts, types.ts)
tests/                 saf məntiq üçün şəbəkəsiz testlər
data/projects/<id>/    source.mp4, work/, output/, jobs/, project.json
```

### Üç fərqli zaman bazası

Hər vaxt sahəsi aid olduğu zaman xəttini adında bildirir:


| baza        | mənası                                                                   |
| ----------- | ------------------------------------------------------------------------ |
| `source_*`  | ilkin yüklənmiş video                                                    |
| `compact_*` | Whisper-ə göndərilən sükutu çıxarılmış səs (transkripsiyanın daxili işi) |
| `output_*`  | son render edilmiş Reel                                                  |


`CompactTimeline.to_source()` ikincidən birinciyə hissə-hissə xətti körpüdür;
`timeline_manager` isə birincidən üçüncüyə körpünü qurur.

### Bilinməsi vacib olan dizayn qərarları

**Yalnız nitqdən ibarət ASR səsi.** VAD seqmentləri yüklənməzdən əvvəl FLAC
parçalarına birləşdirilir. Whisper nəfəs səslərini və ölü havanı heç vaxt
eşitmir (daha az uydurma boşluq tokeni, daha kiçik yükləmə), söz vaxt kodları
isə sonradan mənbə zaman xəttinə geri xəritələnir. Hər birləşmə yerində 4 ms
sönmə klik artefaktlarının qarşısını alır.

**Vaxt kodları heç vaxt LLM tərəfindən dəyişdirilmir.** Terminologiya keçidi
ciddi şəkildə token-token əvəzləmədir: yalnız mövcud tokenin zahiri formasını
dəyişə bilər. Əlavə etmə, silmə, birləşdirmə, bölmə və yenidən vaxtlama
prompt-da xahiş olunmaqla deyil, **kodda** rədd edilir
(`_acceptable_replacement`). ASR-in ilkin yazılışı `Word.raw_text`-də saxlanılır
və UI-də nöqtəli alt xətlə göstərilir.

**Kəsim yalnız söz silinən yerdə baş verir.** Sağ qalan söz zəncirinin içindəki
təbii fasilələr qorunur, buna görə danışıq ritmini saxlayır. Bundan ayrı olaraq
`trim_silences` VAD-ın tapdığı ≥ `SILENCE_TRIM_MS` ölü havanı silir.
`MIN_REMOVABLE_GAP_MS`-dən az qazanc verən silmələr gözlə görünən jump cut-a
çevrilmək yerine birləşdirilir və heç bir klip `MIN_CLIP_MS`-dən qısa olmur.

**Səs birləşmələri zaman xəttini dəqiq qoruyur.** Kəskin kəsim klik verir.
Defolt `crossfade` rejimi hər klipin *səs* trim-ini video kəsim nöqtəsindən `d`
ms sonraya uzadır və `acrossfade=d` ilə birləşdirir; `len(A+d) + len(B) − d == len(A) + len(B)` olduğuna görə ümumi uzunluq dəyişmir, deməli altyazı vaxtları
etibarlı qalır. Keçid materialı kəsimdən dərhal sonrakı real səsdir — yəni
kəskin görüntü kəsimi altında klassik səs keçidi. Bir birləşmə nöqtəsində artıq
material olmadıqda (fayl sonunda bitən kəsim) renderer avtomatik olaraq
klip-başına mikro sönmələr + kəskin concat-a keçir (`AUDIO_JOIN_MODE=fade`).
Yoxlanılıb: iki rejim də hesablanmış müddəti millisaniyə dəqiqliyi ilə təkrarlayır.

**Tək ffmpeg keçidi.** Kəsim → concat → 1080×1920-ə normallaşdırma (örtən
miqyaslama + mərkəzdən kəsim) → B-roll örtükləri → altyazının yandırılması —
hamısı tək bir `filter_complex` daxilində baş verir, buna görə video yalnız bir
dəfə kodlaşdırılır. Yaradılan qraf sazlama üçün
`data/projects/<id>/work/filter_complex.txt` faylına yazılır.

**B-roll dialoqu saxlayır.** Örtüklər yalnız görüntünü dəyişir; stok klipin öz
səsi heç vaxt map edilmir. Üst-üstə düşən örtüklər determinik həll olunur
(əvvəlki qazanır, sonrakı geriyə itələnir və ya atılır).

## 5. Altyazılar: iki mühərrik

İkisi də eyni yerləşdirmə modelini paylaşır (sətirlərə bölmə, söz-başına
vurğulama vəziyyətləri).

- **`ass`** — FFmpeg-in `ass` filtri ilə yandırılan Advanced SubStation Alpha
skripti. Real şrift formalaşdırması, aktiv sözdə animasiyalı böyümə,
sub-piksel konturlar. **libass** tələb edir.
- **`pillow`** — hər vurğulama vəziyyəti üçün şəffaf PNG rasterləşdirir və onları
concat demuxer vasitəsilə tək bir `overlay` ilə birləşdirir. FFmpeg build-ində
libass/libfreetype olmadıqda avtomatik alternativ. Hər söz *vurğulanmış*
ölçüsünə uyğun sahə tutur, buna görə vurğu irəlilədikcə sətir nə titrəyir, nə
də sözlər üst-üstə düşür.

`SUBTITLE_ENGINE=auto` (defolt) mövcud olduqda `ass`, əks halda `pillow` seçir.
`/api/capabilities` hansının seçildiyini və səbəbini bildirir.

Şriftlər adi macOS/Linux/Windows şrift qovluqlarından avtomatik tapılır
(Montserrat ExtraBold → Anton → Inter/Roboto/Noto/DejaVu Bold → Arial Bold →
hər hansı qalın şrift). `SUBTITLE_FONT_FILE=/yol/Font.ttf` ilə əvəz edin.

**Hərf dəstəyi güman edilmir, yoxlanılır.** `Impact` və `Arial Black` şriftlərində
`ə` (U+0259) hərfi üçün qlif yoxdur, deməli yalnız qalınlığa görə seçim hər
azərbaycanca `ə` hərfini boş kvadrat kimi göstərərdi. `font_covers()` funksiyası
`ə ğ ı ş ç ö ü` hərflərinin hər birini render edib bitmap-i `.notdef` qlifi ilə
müqayisə edir, buna görə həmin şriftlər yalnız başqa heç nə quraşdırılmayıbsa
seçilir — nəticə isə `/api/capabilities.font_covers_language` vasitəsilə UI-yə
çatdırılır. Yoxlama yalnız Pillow istifadə edir (əlavə asılılıq yoxdur) və
fontTools-un cmap cədvəlləri ilə müqayisə edilərək təsdiqlənmişdir.

## 6. Azərbaycan dili üçün hansı model?

### İlk modelin niyə sıradan çıxması

`llama-3.3-70b-versatile` Groq tərəfindən **2026-06-17-də köhnəlmiş elan edilib
və 2026-08-16-da tamamilə dayandırılıb** — `llama-3.1-8b-instant`,
`qwen/qwen3-32b` və `meta-llama/llama-4-scout-17b-16e-instruct` ilə birlikdə.
Groq-un özünün tövsiyə etdiyi əvəz `openai/gpt-oss-120b` və ya
`qwen/qwen3.6-27b`-dir. Bu dayandırma pulsuz və developer səviyyələrinə aiddir;
öhdəlikli korporativ hesablara təsir etmir. Məhz buna görə tətbiq artıq tək
sabit model ID-sinə güvənmir, fallback zənciri gəzir.

### İddia deyil, ölçmə

Qalan iki namizəd geniş çoxdilli dəstək iddia edir (Qwen3: türk dilləri
ailəsi daxil olmaqla 119 dil; gpt-oss: 81+ dil), lakin heç biri Azərbaycan dili
üzrə rəqəm açıqlamır və istifadəçi rəyləri gpt-oss-un bəzi az-resurslu
dillərdə zəif tərcümə keyfiyyətini qeyd edir. Buna görə seçim **tətbiqin öz
prompt-u ilə** ölçüldü: 35 azərbaycanca token — düzəldilməli 13 fonetik
təhrif olunmuş ingilis termini, toxunulmamalı 22 düzgün azərbaycanca söz
(`alqoritm`, `marketinq`, `kontent` kimi alınmalar daxil) — hər model 3 dəfə.


| Model                     | düzəldilən /13 | korlanan /22 | orta gecikmə |
| ------------------------- | -------------- | ------------ | ------------ |
| **`openai/gpt-oss-120b`** | **13.0**       | **0**        | 1.1 s        |
| `openai/gpt-oss-20b`      | 13.0           | 0            | 0.7 s        |
| `groq/compound-mini`      | 13.0           | 0            | 3.6 s        |
| `qwen/qwen3.6-27b`        | 12.0           | 0            | 0.6 s        |


Dördü də iki tokenli `prompt engineering` halını tokenləri qaydasız
birləşdirmədən həll etdi. *B-roll* tapşırığında `gpt-oss-120b` ən işlək
ingilis axtarış sorğularını verdi (`"video editing timeline with b-roll"`),
`gpt-oss-20b` ən ümumi olanı (`"storytelling graphic"`), `compound-mini` isə
bir dəfə məna baxımından səhv düzəliş qaytardı (`vörkflou → framework`).

**Qərar:** əsas `openai/gpt-oss-120b`, sonra `openai/gpt-oss-20b` (eyni dəqiq
və daha sürətli, yaradıcı sorğularda zəif), sonra `qwen/qwen3.6-27b` (ən
sürətli, ardıcıl olaraq bir termini buraxır), sonra `groq/compound-mini`
(düzgün, lakin 3 dəfə yavaş). `GROQ_LLM_MODEL` + `GROQ_LLM_FALLBACKS` məhz
budur.

### Benchmark-ın aşkarladığı iki real səhv

1. **Lüğət düzgün azərbaycancanı korlayırdı.** Siyahıda `algoritm` var idi,
 buna görə *bütün* modellər düzgün azərbaycanca `alqoritm` sözünü ona
 "düzəldirdi". Artıq `DEFAULT_GLOSSARY` (ingilis terminləri) ilə
 `AZ_PRESERVE` (azərbaycanca yazılışı artıq düzgün olan alınmalar) ayrıdır
 və prompt-un 7-ci qaydası onları ingilisləşdirməyi qadağan edir. Korlanma
 1/22-dən **bütün modellərdə 0/22-ə** düşdü.
2. **Düşüncə tokenləri bütün büdcəni yeyirdi.** Qwen3 `<think>…</think>` bloklarını
 birbaşa `content` içində qaytarır; `max_tokens=2048` ilə o heç vaxt JSON-a
 çatmır və yararsız cavab verirdi — bu "Qwen azərbaycancada pisdir" kimi
 görünürdü, halbuki sırf bizim səhvimiz idi. `strip_reasoning()` və
 `reasoning_kwargs()` ilə həll olundu (Qwen üçün `reasoning_effort=none`,
 gpt-oss üçün `low`). Diqqət: sadəcə `max_tokens`-i artırmaq həll **deyil** —
 8192 bu hesab səviyyəsində HTTP 413 qaytarır.

### ASR: turbo deyil, `whisper-large-v3` saxlanılır

Azərbaycan dili üçün model ölçüsü həqiqətən əhəmiyyətlidir — dərc olunmuş
FLEURS-az nəticələri:


| Model                | WER        | CER       |
| -------------------- | ---------- | --------- |
| **Whisper large-v3** | **21.7 %** | **5.9 %** |
| MMS-1B (azj-latin)   | 23.8 %     | 5.3 %     |
| Whisper medium       | 34.3 %     | 9.0 %     |
| Whisper small        | 51.7 %     | 14.4 %    |


`whisper-large-v3-turbo` dekoderi 32 təbəqədən 4-ə qədər azaldır və \~4× sürət
üçün \~1–2 % WER verir, ən çox da az-resurslu dillərdə pisləşir — burada bu
mübadilə səhvdir, buna görə `GROQ_ASR_MODEL` `whisper-large-v3` qalır.

Kodda artıq nəzərə alınmış iki azərbaycana xas nəticə:

- **Türk dilinə yönləndirmə.** Avtomatik dil təyini ilə Whisper azərbaycanca
səsi tez-tez türk dilinə yönləndirir və sistemli orfoqrafik səhvlər yaradır.
Tətbiq həmişə açıq şəkildə `language="az"` göndərir (`ASR_LANGUAGE`) — bu,
sənədləşdirilmiş həll yoludur.
- **Bilinən səhv profili** (azərbaycanca): xüsusi adlar 24.1 %, rəqəmlər
22.9 %, düşən `ə ğ ı ö ş ü ç` hərfləri 18.9 %. Terminologiya keçidi yalnız
birinci qrupa kömək edir, deməli xüsusi adlar və rəqəmlər render-dən əvvəl
hələ də insan nəzarəti tələb edir.

## 7. Problemlərin həlli

**`No such filter: 'ass'` / altyazılar gözlədiyinizdən fərqli görünür.**
FFmpeg-inizdə libass yoxdur. `/api/capabilities`-ə baxın. Ya FFmpeg-i libass ilə
yenidən quraşdırın (`brew reinstall ffmpeg`), ya da avtomatik `pillow`
mühərrikini saxlayın — ikisi də yandırılmış kinetik altyazı verir.

**`The model 'x' does not exist or you do not have access to it`.**
Groq model ID-lərini dövri olaraq ləğv edir. Əvvəlcə `GROQ_LLM_MODEL`, sonra
`GROQ_LLM_FALLBACKS`-dəki hər ID sınanır, buna görə tətbiq işləməyə davam edir;
`GROQ_LLM_MODEL`-i hesabınız üçün
`GET https://api.groq.com/openai/v1/models` nəyi qaytarırsa ona qoyun.
Transkripsiya (`whisper-large-v3`) ilə B-roll/terminologiya bir-birindən
asılı deyil — LLM əlçatan olmasa, B-roll an seçimi determinik informasiya
sıxlığı evristikasına keçir və transkript yenə də çatdırılır. Azərbaycan dili üzrə ölçülmüş sıralama üçün §6-ya baxın.

**Sorğu limitləri.** Groq və Pexels 429 cavabları eksponensial gecikmə ilə
təkrarlanır və `Retry-After`-a hörmət edir. Davam etsə, API `429` və
`error.code = upstream_rate_limited` ilə `error.retry_after` qaytarır.

**"Silero VAD could not be loaded".** Sizin Python versiyanız üçün PyTorch
quraşdırılmayıb. Python 3.10–3.13-də `requirements-vad.txt`-i quraşdırın. O
vaxta qədər daxili enerji əsaslı VAD (adaptiv səs döşəməsi + histerezis)
istifadə olunur və layihə xəbərdarlıq göstərir.

**"The uploaded file contains no video/audio stream".** Sözün həqiqi mənasında:
`ffprobe` ikisindən yalnız birini tapdı. Hər iki treklə yenidən ixrac edin.

**macOS: `pip` `Symbol not found: _XML_SetAllocTrackerActivationThreshold` ilə xəta verir.**
Homebrew-un Python-u `pyexpat`-ı sistem `libexpat`-ına bağlayır, macOS 26-da isə
o simvol yoxdur; bu `venv`/`pip`-i sındırır (bu tətbiqi deyil). Həlli:

```bash
brew install expat
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib   # yalnız pip/venv üçün lazımdır
```

Tətbiqin özü `pyexpat` import etmir, buna görə serveri işlədəndə bu dəyişənə
**ehtiyac yoxdur**.

## 8. Konfiqurasiya

`.env.example`-dəki hər dəyər `app/config.py`-dəki `Settings` sahəsinə birbaşa
uyğun gəlir. Ən çox dəyişdirəcəyiniz olanlar:


| Dəyişən                      | Defolt                                               | Təsiri                                                  |
| ---------------------------- | ---------------------------------------------------- | ------------------------------------------------------- |
| `ASR_LANGUAGE`               | `az`                                                 | Whisper dil göstərişi.                                  |
| `ASR_PROMPT`                 | Azərbaycanca + texnologiya/marketinq termin siyahısı | ASR-i sizin lüğətinizə yönəldir.                        |
| `VAD_PAD_MS`                 | `65`                                                 | Hər nitq sərhədi ətrafında əlavə boşluq.                |
| `SILENCE_TRIM_MS`            | `350`                                                | Kəsilməyə dəyən minimum sükut.                          |
| `CUT_PAD_MS`                 | `40`                                                 | Saxlanan söz zəncirinin əvvəl/sonunda buraxılan pay.    |
| `AUDIO_JOIN_MODE`            | `crossfade`                                          | `crossfade` və ya `fade`.                               |
| `SUBTITLE_HIGHLIGHT_COLOR`   | `#FFE94D`                                            | Aktiv sözün rəngi (UI-də render-başına da seçilir).     |
| `SUBTITLE_POSITION`          | `lower_third`                                        | `lower_third` və ya `center`.                           |
| `VIDEO_CRF` / `VIDEO_PRESET` | `20` / `veryfast`                                    | Keyfiyyət/sürət balansı.                                |
| `LOG_LEVEL`                  | `INFO`                                               | Loq səviyyəsi; `DEBUG` ffmpeg filtergraph-ını da yazır. |


## 9. API


| Metod    | Yol                                       | Təyinatı                                                           |
| -------- | ----------------------------------------- | ------------------------------------------------------------------ |
| `GET`    | `/api/health`                             | Canlılıq yoxlaması.                                                |
| `GET`    | `/api/capabilities`                       | FFmpeg build-i, seçilmiş altyazı mühərriki, açar/konfiq vəziyyəti. |
| `POST`   | `/api/projects`                           | Multipart yükləmə; emalı başladır. → `Project`                     |
| `GET`    | `/api/projects`                           | Son layihələr.                                                     |
| `GET`    | `/api/projects/{id}`                      | Transkript və B-roll ilə tam layihə.                               |
| `DELETE` | `/api/projects/{id}`                      | Layihəni və bütün artefaktları silir.                              |
| `GET`    | `/api/projects/{id}/job`                  | Son emal tapşırığı (progress üçün).                                |
| `GET`    | `/api/projects/{id}/timeline`             | Saxlanmış redaktə üçün zaman xətti.                                |
| `PUT`    | `/api/projects/{id}/edit`                 | Xətlənmiş sözləri yazır → `{project, timeline}`.                   |
| `POST`   | `/api/projects/{id}/broll/{sid}/search`   | Yeni sorğu ilə Pexels-də təkrar axtarış.                           |
| `POST`   | `/api/projects/{id}/broll/{sid}/decision` | `accepted` / `dismissed` / `pending`.                              |
| `POST`   | `/api/projects/{id}/render`               | Render başladır → `Job`.                                           |
| `GET`    | `/api/jobs/{job_id}`                      | Tapşırığın vəziyyəti/progress/xətası.                              |
| `GET`    | `/api/projects/{id}/source`               | Mənbə axını (HTTP Range dəstəklənir).                              |
| `GET`    | `/api/projects/{id}/output`               | Render edilmiş axın (Range dəstəklənir).                           |
| `GET`    | `/api/projects/{id}/download`             | Render edilmiş fayl attachment kimi.                               |


Xətalar həmişə `{"error": {"code", "message", "detail?", ...}}` formasındadır və
sabit `code` daşıyır (`missing_audio_stream`, `no_speech_detected`,
`upstream_rate_limited`, `invalid_edit`, `ffmpeg_capability_missing`, …), buna
görə UI mətn müqayisəsi etmədən reaksiya verə bilir.

## 10. Testlər

```bash
pip install pytest
pytest tests -q
```

`tests/` şəbəkəsizdir: zaman xətti hesablamaları, sükut kəsimi, altyazı
yaradılması/escape-i, `.ass` etibarlılığı və `ffprobe` xəta yolları (öz
fixture-larını FFmpeg ilə özü yaradır). API/render inteqrasiya yoxlamaları canlı
Groq və Pexels açarları tələb edir və `tests/README.md`-də sənədləşdirilib.

## 11. Lisenziya / istinad

Stok görüntülər Pexels-dən
[Pexels lisenziyası](https://www.pexels.com/license/) altında gəlir; `BrollClip`
`author`, `author_url` və `page_url` sahələrini daşıyır, buna görə müəllifləri
qeyd edə bilərsiniz.
