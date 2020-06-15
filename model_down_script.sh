mkdir models
cd models
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT' -O ctdet_coco_dla_2x.pth

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1udpOPum8fJdoEQm6n0jsIgMMViOMFinu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1udpOPum8fJdoEQm6n0jsIgMMViOMFinu" \
-O all_dla34.pth && rm -rf /tmp/cookies.txt