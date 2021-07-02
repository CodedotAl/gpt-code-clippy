in_file=$1
language=$2
cat $in_file | xargs -P16 -n1 -I% bash -c 'echo %; \
 name=$(echo % | cut -d"/" -f2); \
 org=$(echo % | cut -d"/" -f1); \
 echo "Cloning $org/$name"
 DIR=Repos/'$language'/$org; \
 mkdir -p $DIR; \
 echo $DIR; \
 git clone -q --depth 1 https://github.com/$org/$name $DIR/$name'