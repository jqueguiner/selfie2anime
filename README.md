This API will convert your selfie into anime drawing.

Providing the sellfie url to the API will returns the conversion of your selfie into an anime.

- - -
Read this paper to learn more about the Image to Image Translation technic:
[https://arxiv.org/abs/1907.10830](https://arxiv.org/abs/1907.10830)

This work is also largely inspired by the [https://selfie2anime.com](https://selfie2anime.com) website made by
[@nathangloverAUS](https://twitter.com/nathangloverAUS)
and 
[@RicoBeti](https://twitter.com/RicoBeti)

```
@article{DBLP:journals/corr/abs-1907-10830,
  author    = {Junho Kim and
               Minjae Kim and
               Hyeonwoo Kang and
               Kwanghee Lee},
  title     = {{U-GAT-IT:} Unsupervised Generative Attentional Networks with Adaptive
               Layer-Instance Normalization for Image-to-Image Translation},
  journal   = {CoRR},
  volume    = {abs/1907.10830},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.10830},
  archivePrefix = {arXiv},
  eprint    = {1907.10830},
  timestamp = {Thu, 01 Aug 2019 08:59:33 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1907-10830},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
- - -
EXAMPLE
![output](https://i.ibb.co/nc8tbcW/example.png)
- - -
INPUT

```json
{
  "url": "https://i.ibb.co/QH0R0DN/input.jpg"
}
```
- - -
EXECUTION FOR DISTANT FILE (URL)
```bash
curl -X POST "https://api-market-place.ai.ovh.net/image-selfie2anime/process" -H "accept: image/png" -H "X-OVH-Api-Key: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX" -H "Content-Type: application/json" -d '{"url":"https://i.ibb.co/QH0R0DN/input.jpg"}'
```
EXECUTION FOR LOCAL FILE (UPLOAD)
```bash
curl -X POST "https://api-market-place.ai.ovh.net/image-selfie2anime/process" -H "accept: image/png" -H "X-OVH-Api-Key: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX" -F file=@input.jpg
```
- - -

OUTPUT

![output](https://i.ibb.co/71KCWnc/output.jpg)

please refer to swagger documentation for further technical details: [swagger documentation](https://market-place.ai.ovh.net/#!/apis/59a0426c-c148-4cff-a042-6cc148fcffa5/pages/ffcac2c8-1c1f-4495-8ac2-c81c1f449524)
