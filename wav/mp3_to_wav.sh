#! /bin/bash
#ffmpeg -i input.mp3 -ab 192k output.wav

for f in *.mp3; do ffmpeg -i "$f" -ab 192k "${f%.mp3}.wav"; done

