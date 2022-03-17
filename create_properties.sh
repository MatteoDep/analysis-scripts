#!/bin/sh

usage(){
    echo "$0 [file]"
    echo "Create json file from table. If file is not given the content of the clipboard is used instead."
}

if [ -z "$1" ]; then
    clip=1
elif [ -f "$1" ]; then
    file="$1"
else
    echo "File $1 does not exist"
    usage
    exit 1
fi

feed_table(){
    if [ $clip ]; then
        xclip -sel clip -o | grep '^S' | tr -d '\r'
    else
        grep '^S' "$file"
    fi
}

lines=$(feed_table | wc -l)

echo '{'
feed_table |
    awk -F '\t' '{
        comma=(NR < '"$lines"'? "," : ""); \
        print \
        "  \"" $1 "\": {\n" \
        "    \"pair\": \"" $2 "\",\n" \
        "    \"injection\": \"" $3 "\",\n" \
        "    \"input\": \"" $4 "\",\n" \
        "    \"output\": \"" $5 "\",\n" \
        "    \"temperature\": \"" $6 "\",\n" \
        "    \"comment\": \"" $7 "\"\n" \
        "  }" comma \
    }'
echo '}'
