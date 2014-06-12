find . -name "*.wav" | while read -r name; do sox "$name" "${name%.*}_TEMP.wav"; done
find . -name "*_TEMP.wav" | while read -r name; do mv "$name" "${name%_TEMP*}.wav"; done

