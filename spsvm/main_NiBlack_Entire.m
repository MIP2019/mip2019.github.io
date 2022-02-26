clear all
close all


path='C:\Mafei\Paper\1-SVM-Patch-LaTeX\1-Reply\Experiments\Original_Image\'
dirFiles=dir(strcat(full(path),'*.tif'))
saveDir=strcat('C:\Mafei\Paper\1-SVM-Patch-LaTeX\1-2-Reply\Q5_AdaptiveThreshold\NiBlack_5\')
if ~exist(saveDir)
    mkdir(saveDir)
end
windows=128
for fi= 1:size(dirFiles,1)
    
    tf=imread(strcat(path,dirFiles(fi).name));    
    imshow(tf)
    tf=double(tf);
    tf=medfilt2(tf,[15,15]);
    gb2=segNiBlack(tf,windows,0.1);
    gb2=medfilt2(gb2,[5,5]);
%     figure();imshow(255-255*gb2,[]);
     imwrite(255-255*gb2,strcat(saveDir,dirFiles(fi).name(1:end-4),'.bmp'))
end
