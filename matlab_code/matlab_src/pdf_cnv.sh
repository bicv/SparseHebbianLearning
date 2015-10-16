#. /sw/bin/init.sh
# ./pdf_conv.sh figures/*.eps
for f in $* ;do
   echo "converting  $f "
   epstopdf --nocompress $f 
#   a2ping --nocompress $f 
done
