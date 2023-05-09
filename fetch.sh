today=`date --iso`
exurl='https://e621.net/db_export/'
yesterday=`date --iso -d yesterday`

for asset in tags posts pools; do
    opath="$asset.csv"
    if [ ! -e "$opath" ]; then 
        curl "$exurl/$asset-$today.csv.gz" | gunzip > "$opath"
        # today's data probably just not dumped yet
        if [ "$?" -ne 0 ]; then
            curl "$exurl/$asset-$yesterday.csv.gz" | gunzip > "$opath"
        fi
    fi
done
