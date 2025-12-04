#!/bin/bash

# --- Konfiguration ---
BIB_FILE="library.bib"
CSL_STYLE="nature.csl"
SOURCE_DIR="content/posts/source"
TARGET_DIR="content/posts"

echo "Starte Pandoc-Konvertierung für Zitate..."
echo "Datenbank: $BIB_FILE | Stil: $CSL_STYLE"
echo "----------------------------------------"

# 1. Sicherstellen, dass die Zielverzeichnisse existieren
mkdir -p "$TARGET_DIR"

# 2. Durchlaufe alle Markdown-Dateien im Quellverzeichnis
for input_file in "$SOURCE_DIR"/*.src.md; do
    # Den Dateinamen ohne Pfad extrahieren
    filename=$(basename "$input_file")

    # Die Zieldatei im Hugo-Content-Ordner definieren
    output_file="$TARGET_DIR/$filename.md"

    echo "Verarbeite: $filename -> $output_file"

    # Pandoc-Befehl ausführen:
    pandoc \
        --citeproc \
        --bibliography="$BIB_FILE" \
        --csl="$CSL_STYLE" \
        -o "$output_file" \
        "$input_file"

    if [ $? -ne 0 ]; then
        echo "FEHLER: Pandoc-Konvertierung für $filename fehlgeschlagen. Abbruch."
        exit 1
    fi
done

echo "----------------------------------------"
echo "Alle Zitate wurden formatiert und in den Content-Ordner verschoben."

# 3. Hugo-Build starten
echo "Starte Hugo-Build..."
hugo

echo "Build abgeschlossen."
