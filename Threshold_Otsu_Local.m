clear;%清除变量的状态数据
clc;%清除命令行
close all
% path='C:\Mafei\1-MachineLearning\Mice\MiceEyeDB_50_45_New\'
% filename='0029 (51).tif'
% % filename='OCT.jpg'
% I=imread(strcat(path,filename));
path='C:\Mafei\1-MachineLearning\Paper\1-Binarization\Figure\Experiments\Original_Image\'

saveDir=strcat('.\LocalThreshold\')
if ~exist(saveDir)
    mkdir(saveDir)
end
dirFiles=dir(strcat(full(path),'410*.tif'));
% path='C:\Mafei\1-MachineLearning\Paper\1-Binarization\Patch_SVM\'
% dirFiles=dir(strcat(full(path),'210_tif_245_1427*.JPG'))
for windows=  32:128:2048
    windows
    for fi= 1:size(dirFiles,1)
        %         fi/size(dirFiles,1)
        filename=dirFiles(fi).name;
        str1=strsplit(filename,'.');
        saveFileName=str1(1);
        I=(imread(strcat(path,filename)));
        ww=size(I,1)+mod(size(I,1),windows)*2+windows ;
        hh=size(I,2)+mod(size(I,1),windows)*2+windows;
        ExpImg=uint8(zeros(ww,hh));
        ExpImg(1:size(I,1),1:size(I,2))=I;
%         imshow(ExpImg);
        img=zeros(ww,hh);
        %          imshow(I,[]);
        %%
        for m=1:floor(size(I,1)/windows)+1
            for n=1:floor(size(I,1)/windows)+1
                x0=(m-1)*windows+1;
                x1=(m)*windows;
                y0=(n-1)*windows+1;
                y1=(n)*windows;
                subImg=ExpImg(x0:x1,y0:y1);
                %                  imshow(subImg,[]);
                bw=graythresh(subImg);
                newII=im2bw(subImg,bw);
                %                 subplot(1,2,2);
                %                 imshow(newII,[]);
                img(x0:x1,y0:y1)=newII*255;
              
            end
        end
        
    end
    imshow(img(1:size(I,1),1:size(I,2)),[]);
    imwrite(img(1:size(I,1),1:size(I,2)),strcat(saveDir,num2str(windows),'_Local_OTSU_',filename(1:end-4),'.bmp'))
end

