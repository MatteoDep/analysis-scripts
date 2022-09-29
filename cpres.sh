#!/bin/sh
# move results in report image directiory and convert to pdf to increase latex compilation speed
# depends on imagemagik (convert command)

report_images_dir='../report/images'
subdirs='data light length temperature gate'

for subdir in $subdirs; do
	sourcedir="./results/$subdir"
	cp -r "$sourcedir" "$report_images_dir/"
	for file in "$report_images_dir/$subdir"/*.png; do
		if [ -f "$file" ]; then
			filenoext=${file%.*}
			convert "$file" -density 100 "$filenoext".pdf
			rm "$file"
		fi
	done
done
# cp -r ./results/data ../report/images/
# cp -r ./results/light ../report/images/
# cp -r ./results/length ../report/images/
# cp -r ./results/temperature ../report/images/
# cp -r ./results/gate ../report/images/
