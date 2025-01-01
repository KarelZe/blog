---
title: Timelapse of writing a thesis in 180 days🎓
date: '2024-01-05T17:35:00+02:00'
Description: Timelapse of writing a thesis
Tags: [academia, thesis, making-of, timelapse]
Categories: [academia]
DisableComments: false
---

While procrastinating during the research phase for my thesis, I came across [a timelapse of a research paper in the making](https://youtu.be/hNENiG7LAnc?feature=shared) by Tim Weninger. I wanted to create a video similar to his but for my own thesis.🥳

## Timelapse

{{< youtube nTiMGg_rock >}}

## Making-of

The basic idea is to create montages from different versions of the pdf and stitch them together as a video. All it needs is a versioned LaTeX document, a bit of [github magic](https://github.com/features/actions), and the power of [ImageMagick](https://imagemagick.org/) and [FFmpeg](https://ffmpeg.org/).

First, I set up a github action in [my thesis repo](https://github.com/KarelZe/thesis), that compiled my document and committed the binary to a second repo. The action runs on every push to the `reports/` directory. Every pushed commit becomes a new frame in the final video.

```yaml
name: Build LaTeX document
on:
  push:
    paths:
      - reports/**
  workflow_dispatch: null
jobs:
  build_latex:
    runs-on: ubuntu-latest
    steps:
      - name: Set up Git repository
        uses: actions/checkout@v4
      - name: Compile LaTeX document
        uses: xu-cheng/latex-action@master
        with:
          latexmk_shell_escape: true
          root_file: |
            thesis.tex
          working_directory: reports
          latexmk_use_lualatex: true
      - name: Release
        uses: softprops/action-gh-release@v1
        if: startsWith(github.ref, 'refs/tags/')
        with:
          files: |
            reports/thesis.pdf
      - name: save tmp copy
        run: |
          mkdir ~/tmplatex
          mv reports/thesis.pdf ~/tmplatex/thesis.pdf
      - name: setup target repo
        uses: actions/checkout@v4
        with:
          repository: KarelZe/thesis2video
          ref: main
          token: ${{secrets.API_TOKEN_GITHUB}}
      - name: setup git config
        run: |
          git config user.name "GitHub Actions Bot"
          git config user.email "action@markusbilz.com"
      - name: Move file and push
        run: |
          mv ~/tmplatex/thesis.pdf ${{github.event.repository.pushed_at}}-thesis.pdf
          git add -A
          git commit -m "Version @ ${{github.event.repository.pushed_at}}"
          git push origin main
```

After I had created the final version of my thesis, I had a second repo (`KarelZe/thesis2video`) full of pdf files versioned by timestamp. As some commits like the removal of unused packages did not affect the layout, I decided to deduplicate files before moving any further.

```bash
find . -type f -exec md5sum {} + | sort | uniq -w32 -d --all-repeated=separate | sed -r 's/^[^ ]* //' | xargs rm
```

To create the [montages](https://imagemagick.org/script/montage.php) from the pdfs, I used the [paper2movie script](https://github.com/momentofgeekiness/paper2movie) by Raymond Vermaas. I tweaked the script to adapt to my preferences. In particular, I adjusted the scaling logic to dynamically scale the size of the tiles in the montage with the number of pages in the document and removed the code for compiling the LaTeX documents and reading the git history, as I had compiled the pdf files already.

Initially, I experimented with converting the pdf to frames using a GitHub action but ran into memory issues quickly. Therefore, I decided to perform the conversion locally.

```shell
#!/bin/bash

# adapted from: https://github.com/momentofgeekiness/paper2movie

############################
### INPUTS ###
############################
# Name of the (input) paper file (without extension)
filename=thesis.avi

# Video size (by default 4k)
totalWidth=3840
totalHeight=2160
# Video frames per second
# Use low values (< 5) on papers with many pages (> 50).
# It will give a better results, since there is much more to look at. ;-)
fps=4
############################
### CLEANUP STUFF ###
############################
echo -e "\nCleaning up...\n"
rm *.png > /dev/null
rm *.avi > /dev/null

############################
### CREATING IMAGES ###
############################
for f in `ls -1 *-thesis.pdf`; do
pages=`pdfinfo $f | awk '/^Pages/ { print $2}'`
# maximum number of pages
# The width/height ratio of A4 paper (1/sqrt(2))
ratioA4="0.7070"
# The height of an individual tile
tileHeight=`echo "scale=10;sqrt(($totalWidth*$totalHeight)/($ratioA4*$pages))" | bc`
tileWidth=`echo "scale=10;$tileHeight*$ratioA4" | bc`
# Calculate grid
numTilesHeight=`echo "scale=10;$totalHeight/$tileHeight" | bc`
numTilesWidth=`echo "scale=10;$totalWidth/$tileWidth" | bc`
# Ceil tiles to integers,
numTilesHeight=`awk -v var="$numTilesHeight" 'BEGIN{var = var < 0 ? int(var) : (int(var) + (var == int(var) ? 0 : 1)); print var}'`
numTilesWidth=`awk -v var="$numTilesWidth" 'BEGIN{var = var < 0 ? int(var) : (int(var) + (var == int(var) ? 0 : 1)); print var}'`
# Report measurements
echo -e "Processing File: $f"
echo -e "\nMovie measurements:"
echo -e "Number of horizontal tiles: $numTilesWidth"
echo -e "Number of vertical tiles: $numTilesHeight"
# Having ceiled the number of tiles, they exceed the totalWidth and totalHeight.
# So, we also need to recalculate the tileHeight and tileWidth. This step will
# (slightly) change the A4 ratio, but it beats half pages in the video.
tileHeight=`echo "scale=5;$totalHeight/$numTilesHeight" | bc`
tileHeight=`echo $tileHeight |cut -f1 -d"."`
tileWidth=`echo "scale=5;$totalWidth/$numTilesWidth" | bc`
tileWidth=`echo $tileWidth |cut -f1 -d"."`

montage $f -tile ${numTilesWidth}x${numTilesHeight} -background white -geometry ${tileWidth}x${tileHeight} $f.png
done
############################
### RENDER MOVIE ###
############################
echo -e "\nRendering movie...\n"
cat $(find . -maxdepth 1 -name "*.png" | sort -V) | ffmpeg -r $fps -i - $filename

echo -e "Movie available at $filename"
```

Depending on the image magick version installed, you might have to [adjust policies for pdf conversion](https://askubuntu.com/a/1081907). That's all it needs.
