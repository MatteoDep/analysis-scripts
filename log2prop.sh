#!/bin/sh

usage(){
    echo "$0 [file]"
    echo "Create json file from table. If file is not given the content of the clipboard is used instead."
}

sep='\t'
append=

while getopts "s:ah" option
do
    case "${option}" in
        s)
            sep="$OPTARG"
            ;;
        a)
            append=1
            ;;
        h)
            usage
            exit
            ;;
        *)
            echo "unsupported option $option."
            usage
            exit 1
            ;;
    esac
done
shift $((OPTIND-1))

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
        xclip -sel clip -o
    else
        cat "$file"
    fi
}

# check if pasted from libreoffice or MS word
modify_lines(){
    if feed_table | grep -q '^[0-9]'; then
        awk 'NR%7{printf("%s\t", $0); next} {print $0}'
    else
        tr -d '\r'
    fi
}


[ -z $append ] && echo "name pair injection input output temperature comment" | tr ' ' "$sep"
feed_table |
    modify_lines |
    grep '^S' |
    tr '\t' "$sep"
