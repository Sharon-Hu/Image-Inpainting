function inpainted = multiImgInpa(baseImg,mask,imgCom1,imgCom2,fillcolor)
%MULTIPLE IMAGES INPAINTIGN
%   for several input images to determine the mask
%  baseI
[img,mask,comimg1,comimg2,fillRegion] = loadimgs(baseImg,mask,imgCom1,imgCom2,fillcolor);
sz=[size(img,1) size(img,2)]; 
imgG = rgb2lab(img); comimg1G = rgb2lab(comimg1); comimg2G = rgb2lab(comimg2);
img=double(img);
fillRini = fillRegion;
origImg = img;
wi=3;
ind = img2ind(img);
sourceRegion = ~fillRegion;
sourceRini = sourceRegion;
ck=1;
fillRegionD = double(fillRegion);  %size与原图相同，二维
  
edgeD = edge(fillRegionD,'canny');
 smo = find(edgeD>0);

while any(fillRegion(:)) && ck<100
    fillRegionD = double(fillRegion);  %size与原图相同，二维
  
  % to determine the fill-front 
  %dR = find(conv2(fillRegionD,[1,1,1;1,-8,1;1,1,1],'same')>0);
  edgeD = edge(fillRegionD,'canny');
   dR = find(edgeD>0);
   ckk=size(dR)
    for j = 1:numel(dR)
        k = dR(j);
       [Hp,rows,cols] = getpatch(sz,k,wi);
       toFill = fillRegion(Hp);
        toFill = logical(toFill);
        delt1 = imgG(Hp(toFill))-comimg1G(Hp(toFill));d1 = find(delt1==0);
        delt2 = imgG(Hp(toFill))-comimg2G(Hp(toFill));d2 = find(delt2==0);
       if  numel(d1)==0
           seeeee=1;
            img(rows,cols,:) = ind2img(ind(rows,cols),comimg1);
        elseif numel(d2)==0
            sddd = 2;
           img(rows,cols,:) = ind2img(ind(rows,cols),comimg2);
%            
       else
            fff =3;
           Hq = bestexemplar(img,img(rows,cols,:),toFill',sourceRegion);
           ind(Hp(toFill)) = ind(Hq(toFill));
           img(rows,cols,:) = ind2img(ind(rows,cols),origImg);  
       end
        fillRegion(Hp(toFill)) = false;
        sourceRegion = ~fillRegion;
   end
     ck=ck+1
end
for i =1:numel(smo)
    k = smo(i);
     [Hp,rows,cols] = getpatch(sz,k,wi);
     toFill = fillRini(Hp);
     toFill=logical(toFill);
%      S = candPa(img(rows,cols,:),2,sourceRini,toFill,img);
%      d = S{1};
%      r = d(1); R = d(2);
%      c = d(3); CO = d(4);
%      ro = r:R; co = c:CO;

 Hq = bestexemplar(img,img(rows,cols,:),toFill',sourceRegion);
           ind(Hp(toFill)) = ind(Hq(toFill));
           img(rows,cols,:) = ind2img(ind(rows,cols),img);
%            ind(Hp(toFill)) = ind(Hq(toFill));
    %       img(rows,cols,:) = ind2img(ind(ro,co),img);
     
end

inpainted = img;
 image(uint8(inpainted));


% 边边上要留出来一点做修补



function [img,mask,comimg1,comimg2,fillRegion] = loadimgs(origFileName,mask,file1,file2,fillColor)
img = imread(origFileName); comimg1 = imread(file1); comimg2 = imread(file2); mask = imread(mask);
fillRegion = mask(:,:,1)==fillColor(1) & mask(:,:,2)==fillColor(2) & mask(:,:,3)==fillColor(3);

%---------------------------------------------------------------------
% Returns the indices for a 9x9 patch centered at pixel p.
%---------------------------------------------------------------------
function [Hp,rows,cols] = getpatch(sz,p,w)
% [x,y] = ind2sub(sz,p);  % 2*w+1 == the patch size
%w=4; 
p=p-1; y=floor(p/sz(1))+1; p=rem(p,sz(1)); x=floor(p)+1;
rows = max(x-w,1):min(x+w,sz(1));
cols = (max(y-w,1):min(y+w,sz(2)))';
Hp = sub2ndx(rows,cols,sz(1));

%---------------------------------------------------------------------
% Converts the (rows,cols) subscript-style indices to Matlab index-style
% indices.  Unforunately, 'sub2ind' cannot be used for this.
%---------------------------------------------------------------------
function N = sub2ndx(rows,cols,nTotalRows)
X = rows(ones(length(cols),1),:);
Y = cols(:,ones(1,length(rows)));
N = X+(Y-1)*nTotalRows;

%---------------------------------------------------------------------
% Converts an indexed image into an RGB image, using 'img' as a colormap
%---------------------------------------------------------------------
function img2 = ind2img(ind,img)
for i=3:-1:1, temp=img(:,:,i); img2(:,:,i)=temp(ind); end;


%---------------------------------------------------------------------
% Converts an RGB image into a indexed image, using the image itself as
% the colormap.
%---------------------------------------------------------------------
function ind = img2ind(img)
s=size(img); 
ind=reshape(1:s(1)*s(2),s(1),s(2));  %给每个pixel编号，编号方法是[1 3;2 4]

%---------------------------------------------------------------------
% Scans over the entire image (with a sliding window)
% for the exemplar with the lowest error. Calls a MEX function.
%---------------------------------------------------------------------
function Hq = bestexemplar(img,Ip,toFill,sourceRegion)
m=size(Ip,1); mm=size(img,1); n=size(Ip,2); nn=size(img,2);
%[l a b] = RGB2Lab(img(:,:,1),img(:,:,2),img(:,:,3));[l1 a1 b1] = RGB2Lab(Ip(:,:,1),Ip(:,:,2),Ip(:,:,3));
best = bestexemplarhelper(mm,nn,m,n,img,Ip,toFill,sourceRegion);
Hq = sub2ndx(best(1):best(2),(best(3):best(4))',mm); 