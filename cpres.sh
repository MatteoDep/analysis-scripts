#!/bin/sh
# move results in report image directiory and convert to pdf to increase latex compilation speed
# depends on imagemagik (convert command)

report_images_dir='../report/images'
subdirs='data light length temperature gate'

for subdir in $subdirs; do
	sourcedir="results/$subdir"
	destdir="$report_images_dir/$subdir"
	for source_file in "$sourcedir"/*; do
		if [ -f "$source_file" ]; then
			dest_file="$destdir/${source_file##*/}"
			dest_base="${dest_file%.*}"
			ext="${source_file##*.}"
			case "$ext" in
				png)
					printf "converting %s to %s.pdf\n" "$source_file" "$dest_base"
					convert "$source_file" -density 100 "$dest_base".pdf
					;;
				tex)
					printf "copying %s to %s\n" "$source_file" "$dest_file"
					cp "$source_file" "$dest_file"
					;;
				csv)
					printf "skipping %s\n" "$source_file"
					;;
				*)
					printf "unrecognized extension of file %s, skipping...\n" "$source_file"
					;;
			esac
		fi
	done
done
