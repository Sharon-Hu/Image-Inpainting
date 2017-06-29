function [inpaintedImg,origImg,fillImg,C,D,fillMovie] = inpaintStable(imgFilename,fillFilename,fillColor)
%INPAINT  Exemplar-based inpainting.
%
% Usage:   [inpaintedImg,origImg,fillImg,C,D,fillMovie] ...
%                = inpainting(imgFilename,fillFilename,fillColor)
% Inputs: 
%   imgFilename    Filename of the original image.
%   fillFilename   Filename of the image specifying the fill region. 
%   fillColor      1x3 RGB vector specifying the color used to specify
%                  the fill region.
% Outputs:
%   inpaintedImg   The inpainted image; an MxNx3 matrix of doubles. 
%   origImg        The original image; an MxNx3 matrix of doubles.
%   fillImg        The fill region image; an MxNx3 matrix of doubles.
%   C              MxN matrix of confidence values accumulated over all iterations.
%   D              MxN matrix of data term values accumulated over all iterations.
%   fillMovie      A Matlab movie struct depicting the fill region over time. 
%
% Example:
%   [i1,i2,i3,c,d,mov] = inpainting('bungee0.png','bungee1.png',[0 255 0]);
%   plotall;           % quick and dirty plotting script
%   close; movie(mov); % grab some popcorn 
%
   

warning off MATLAB:divideByZero
%[img,fillImg,fillRegion] = loadimgs(imgFilename,fillFilename,fillColor);
%img = double(img);  
%[img,fillImg,fillRegion] = loadimgs(imgFilename,fillFilename,fillColor);
img = double(imgFilename); fillImg = double(fillFilename);
ckk=size(fillImg)
fillRegion = (fillImg(:,:,1)==fillColor(1)) & (fillImg(:,:,2)==fillColor(2)) & (fillImg(:,:,3)==fillColor(3));
check=find(fillRegion>0);
%return
origImg = img;
ind = img2ind(img);
sz = [size(img,1) size(img,2)];  %返回图像的尺寸，这里的1 2 表示这个矩阵的第几dimension
sourceRegion = ~fillRegion;  %
w=4;
% Initialize isophote values
% [Ix(:,:,3) Iy(:,:,3)] = gradient(img(:,:,3));
% [Ix(:,:,2) Iy(:,:,2)] = gradient(img(:,:,2));
% [Ix(:,:,1) Iy(:,:,1)] = gradient(img(:,:,1));
% Ix = sum(Ix,3)/(3*255); Iy = sum(Iy,3)/(3*255);
% temp = Ix; Ix = -Iy; Iy = temp;  % Rotate gradient 90 degrees

%
  [Ix(:,:,3) Iy(:,:,3)] = gradient(img(:,:,3));
  [Ix(:,:,2) Iy(:,:,2)] = gradient(img(:,:,2));
  [Ix(:,:,1) Iy(:,:,1)] = gradient(img(:,:,1));
  Ix = sum(Ix,3)/(3*255); Iy = sum(Iy,3)/(3*255);
  temp = Ix; Ix = -Iy; Iy = temp;

% [l a b] = RGB2Lab(img(:,:,1),img(:,:,2),img(:,:,3));
% [Ix(:,:,3) Iy(:,:,3)] = gradient(l);
% [Ix(:,:,2) Iy(:,:,2)] = gradient(a);
% [Ix(:,:,1) Iy(:,:,1)] = gradient(b);
% 
% Ix = sum(Ix,3)/(1400); Iy = sum(Iy,3)/(1400);
% bias = ones(size(Ix(:,:,2)));
% Ix = (Ix(:,:,3)./100 +(Ix(:,:,2)+bias.*500)./1000 + (Ix(:,:,1)+bias.*200)./400)./3;
%   Iy = (Iy(:,:,3)./100 +(Iy(:,:,2)+bias.*500)./1000 + (Ix(:,:,1)+bias.*200)./400)./3;
% temp = Ix; Ix = -Iy; Iy = temp;

% Initialize confidence and data terms
C = double(sourceRegion);
D = repmat(.001,sz);  %构建一个由-0.1形成的与图像尺寸相同的矩阵
iter = 1;
% Visualization stuff
if nargout==6
  fillMovie(1).cdata=uint8(img); 
  fillMovie(1).colormap=[];
  origImg(1,1,:) = fillColor;
  iter = 2;
end

% Seed 'rand' for reproducible results (good for testing)
rand('state',0);
ck=1;
% Loop until entire fill region has been covered

while any(fillRegion(:))  %返回的是一个一维的只有0和1的array，长度与原图的第二维相同
  % Find contour & normalized gradients of fill region
  fillRegionD = double(fillRegion);  %size与原图相同，二维
  
  % to determine the fill-front 
  %dR = find(conv2(fillRegionD,[1,1,1;1,-8,1;1,1,1],'same')>0);
   edgeD = edge(fillRegionD,'canny');
   dR = find(edgeD>0);
  %dR = find(conv2(fillRegionD,[
  imgGray = rgb2gray(img);
 sourceRegion=~fillRegion;
%--------------------------------------------------------------------------
% to calculate the normal of the contour
%--------------------------------------------------------------------------
[row, col]=ind2sub(size(fillRegionD),dR(:));
% figure
% plot(row(1:30),col(1:30),'*')
Nx = zeros(length(dR),1);
Ny = zeros(length(dR),1);
N(1,:) = [0 0];
if length(row)==1
    Nx=1;
    Ny=1;
    N = [Nx Ny];
else
   for h=2:length(row)-1
    Nx(h) = 1 ; %col(h+1)-col(h-1);
    Ny(h) = 1; %row(h-1)-row(h+1);
   end
 N = [Nx Ny];
 N(1,:)=N(2,:);
 N(length(dR),:) = N(length(dR)-1,:); 
end

 N = normr(N);
 N(~isfinite(N))=0; % handle NaN and Inf
 
 
%-------------------------------------------------------------------------
% calculating isophote
%-------------------------------------------------------------------------
% Ix = conv2(img(:,:,1),[0 1 2;-1 0 1;-2 -1 0],'same');
% Iy = conv2(img(:,:,1),[-2 -1 0;-1 0 1;0 1 2],'same');

  
  % Compute confidences along the fill front
  for j=1:length(dR)
     k=dR(j);
    Hp = getpatch(sz,k,w);
    q = Hp(~(fillRegion(Hp)));
    C(k) = sum(C(q))/numel(Hp);
  

  end
  
  % Compute patch priorities = confidence term * data term
 
  D(dR) = abs(Ix(dR).*N(:,1)+Iy(dR).*N(:,2))+ 0.001;
  
  priorities =  C(dR).*D(dR);
  %priorities =  D(dR);
 
 
  % Find patch with maximum priority, Hp
  [unused,ndx] = max(priorities(:));
  p = dR(ndx(1));
  [Hp,rows,cols] = getpatch(sz,p,w);
  toFill = fillRegion(Hp);
 % toLook = sourceRegion(Hp)
  
  % Find exemplar that minimizes error, Hq
  Hq = bestexemplar(img,img(rows,cols,:),toFill',sourceRegion);
  
  % Update fill region
  toFill = logical(toFill);                 % Marcel 11/30/05
  fillRegion(Hp(toFill)) = false;
  
  % Propagate confidence & isophote values
  C(Hp(toFill))  = C(p);
    Ix(Hp(toFill)) = Ix(Hq(toFill));
  Iy(Hp(toFill)) = Iy(Hq(toFill));

  
  % Copy image data from Hq to Hp
  ind(Hp(toFill)) = ind(Hq(toFill));
  img(rows,cols,:) = ind2img(ind(rows,cols),origImg);  
  ck=ck+1


  % Visualization stuff
  if nargout==6
    ind2 = ind;
    ind2(logical(fillRegion)) = 1;          % Marcel 11/30/05
    %ind2(fillRegion) = 1;                  % Original
    fillMovie(iter).cdata=uint8(ind2img(ind2,origImg)); 
    fillMovie(iter).colormap=[];
%    imwrite(rgb2ind(fillMovie(iter).cdata,256),te,'ddd.gif','WriteMode','Append');
  end
  iter = iter+1;
  
  [Ix(:,:,3) Iy(:,:,3)] = gradient(img(:,:,3));
  [Ix(:,:,2) Iy(:,:,2)] = gradient(img(:,:,2));
  [Ix(:,:,1) Iy(:,:,1)] = gradient(img(:,:,1));
  Ix = sum(Ix,3)/(3*255); Iy = sum(Iy,3)/(3*255);
  temp = Ix; Ix = -Iy; Iy = temp;
  sourceRegion=~fillRegion;
% [l a b] = RGB2Lab(img(:,:,1),img(:,:,2),img(:,:,3));
% [Ix(:,:,3) Iy(:,:,3)] = gradient(l);
% [Ix(:,:,2) Iy(:,:,2)] = gradient(a);
% [Ix(:,:,1) Iy(:,:,1)] = gradient(b);
%  Ix = sum(Ix,3)/(1400); Iy = sum(Iy,3)/(1400);
% % Ix = (Ix(:,:,3)./100 +(Ix(:,:,2)+bias.*500)./1000 + (Ix(:,:,1)+bias.*200)./400)./3;
% % Iy = (Iy(:,:,3)./100 +(Iy(:,:,2)+bias.*500)./1000 + (Ix(:,:,1)+bias.*200)./400)./3;
% temp = Ix; Ix = -Iy; Iy = temp;


end

inpaintedImg=img;
image(uint8(inpaintedImg));


%---------------------------------------------------------------------
% Scans over the entire image (with a sliding window)
% for the exemplar with the lowest error. Calls a MEX function.
%---------------------------------------------------------------------
function Hq = bestexemplar(img,Ip,toFill,sourceRegion)
m=size(Ip,1); mm=size(img,1); n=size(Ip,2); nn=size(img,2);
%[l a b] = RGB2Lab(img(:,:,1),img(:,:,2),img(:,:,3));[l1 a1 b1] = RGB2Lab(Ip(:,:,1),Ip(:,:,2),Ip(:,:,3));
best = bestexemplarhelper(mm,nn,m,n,img,Ip,toFill,sourceRegion);
Hq = sub2ndx(best(1):best(2),(best(3):best(4))',mm); 


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
% Loads the an image and it's fill region, using 'fillColor' as a marker
% value for knowing which pixels are to be filled.
%---------------------------------------------------------------------
function [img,fillImg,fillRegion] = loadimgs(imgFilename,fillFilename,fillColor)
img = imread(imgFilename); fillImg = imread(fillFilename);
fillRegion = fillImg(:,:,1)==fillColor(1) & fillImg(:,:,2)==fillColor(2) & fillImg(:,:,3)==fillColor(3);
% fillRegion = fillImg(:,:,1)==fillColor(1) & ...
%     fillImg(:,:,2)==fillColor(2) & fillImg(:,:,3)==fillColor(3);

%----------------------------------------------------------------------
% To find the max. gradient in the patch as the isophote
%----------------------------------------------------------------------


function [A] = normr(N)
for ii=1:size(N,1)
    A(ii,:) = N(ii,:)/norm(N(ii,:));
end









