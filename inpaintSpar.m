function [inpaintedImg,origImg,fillImg,C,sp,fillMovie] = inpaintSpar(origFileName,fillReFileN,fillColor)
% INPIANT SPARSITY  inpainting with patch sparsity
%
% Usage:   [inpaintedImg,origImg,fillImg,C,sp,fillMovie] ...
%                = inpaintSpar(origFilename,fillReFileN,fillColor)
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
%   [i1,i2,i3,c,s,mov] = inpainting('bungee0.png','bungee1.png',[0 255 0]);
%   plotall;           % quick and dirty plotting script
%   close; movie(mov); % grab some popcorn 
%
% Special Notice: patch size 7x7; window size 51x51; epsi
%warning off MATLAB:divideByZero
warning off all
[img,fillImg,fillRegion] = loadimgs(origFileName,fillReFileN,fillColor);
imglabG = rgb2lab(img);
imglabG = imglabG(:,:,1);
imglabG = double(imglabG);
img = double(img);
imglab=img;

origImg = img;
orilab = imglab;
ind = img2ind(img);  % 每个pixel都有它的编号，编号方法与index相同
sz = [size(img,1) size(img,2)];  %返回图像的尺寸，这里的1 2 表示这个矩阵的第几dimension
sourceRegion = ~fillRegion;  %
wi = 4;  % patch size
Ni = 25;  % neighourhood window size
epsi = 0.2; % to set esip for linear transformation of patch sparsity
T = 25;
ck = 1;
% -----------------------------------------------------------
% Initialization

% Initialize confidence and data terms
C = double(sourceRegion);
sp = repmat(0.200,sz);  % initialization of sp

iter = 1;

% Visualization stuff
% Visualization stuff
if nargout==6
  fillMovie(1).cdata=uint8(img); 
  fillMovie(1).colormap=[];
  origImg(1,1,:) = fillColor;
  iter = 2;
end

% Seed 'rand' for reproducible results (good for testing)
rand('state',0);

while any(fillRegion(:))   % 循环以填补
    
    % Find contour & normalized gradients of fill region
    fillRegionD = double(fillRegion);  %size与原图相同，二维
 
    % to determine the fill-front
   % dR = find(conv2(fillRegionD,[1,1,1;1,-8,1;1,1,1],'same')>0);  %lapalacian算子确定
   edgeD = edge(fillRegionD,'canny');
   dR = find(edgeD>0);
  imglabG = RGB2Lab(uint8(imglab));
  imglabG = imglabG(:,:,1);
    for gi=1:numel(dR)                                % for every point in the fill-front

        k = dR(gi);
%         if sp(k)~=0.200
%             continue;
%         end
        
        Hp = getpatch(sz,k,wi);
        if (numel(Hp)<((2*wi+1)^2))
            dR(gi)=0;
% %             if numel(dR)==0
% %                 wi = wi-1;
% %             end
            continue %
        end
        q = Hp(~(fillRegion(Hp)));

        C(k) = sum(C(q))/numel(Hp);
        
        % to calculate sparsity
        Np = getNei(sz,k,Ni);  % 求window的位置

        [spar,upper,w] = sparsity(Hp,Np,fillRegion,sourceRegion,sz,imglabG,imglab,wi); % 计算patch sparsity
        if w==0
            continue;
        end
        % linear transformation to make interval [epsi,1]
        sp(k) = (1-epsi)/(upper-sqrt(1/numel(Np)))*spar + (upper*epsi-sqrt(1/numel(Np)))/(upper-sqrt(1/numel(Np)));
        
    end
    dR(dR==0)=[];
     ck=ck+1
    if numel(dR)==0
        wi = wi-1;
        if wi==0
            break;
        else
            continue;
        end
    end
    priorities = C(dR).*sp(dR);
    
%     if ck==100
%         break;
%     end
        
    
    
    % Find patch with maximum priority, Hp
    [unused,ndx] = max(priorities(:));
    p = dR(ndx(1));    % index may get over the patch size, resulting error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    [Hp,rs,cs] = getpatch(sz,p,wi);
    
    Np = getNei(sz,p,Ni);
    
    [spar,upper,w,fi] = sparsity(Hp,Np,fillRegion,sourceRegion,sz,imglabG,imglab,wi);
   checkfi = numel(fi);
    toFill = fillRegion(Hp);
    
    candi = candPa(imglab(rs,cs,:),T,sourceRegion,toFill,imglab);
   
    
    [AL,S] = LLE(candi,fi,Hp,imglab,sourceRegion);

    toFill = logical(toFill);               
  
    C(Hp(toFill)) = C(p);
   
    % Use all patches in S to fill the region
    d = S{1};
    r = d(1); R = d(2);
    c = d(3); CO = d(4);
    ro = r:R; co = c:CO;
    imfi = ind2img(ind(ro,co),imglab).*AL(1);   % The first patch to fill the region
    
    for ki = 2:numel(S)
        hh = S{ki};
        r = hh(1);R = hh(2);
        c = hh(3);CO = hh(4);
        ro = r:R; co = c:CO;
        imfi = imfi + ind2img(ind(ro,co),imglab).*AL(ki);%.*AL(ki);
    end
    
    % Copy the estimated image to the unknown region
    for cc = 3:-1:1
        tem = imfi(:,:,cc);
        Ce = imglab(:,:,cc);
        Ce(Hp(toFill)) = tem(toFill);
        imglab(:,:,cc) = Ce;
    end
    
    % Update the region to be inpainted
     fillRegion(Hp(toFill)) = false;
     sourceRegion = ~fillRegion;
    
    % Visualization stuff
    if nargout==6
    ind2 = ind;
    ind2(logical(fillRegion)) = 1;          % Marcel 11/30/05
    %ind2(fillRegion) = 1;                  % Original
    fillMovie(iter).cdata=uint8(ind2img(ind2,imglab)); 
    fillMovie(iter).colormap=[];
%    imwrite(rgb2ind(fillMovie(iter).cdata,256),te,'ddd.gif','WriteMode','Append');
  end
  iter = iter+1;

end
inpaintedImg = imglab; % Lab2RGB(imglab); %,'OutputType','uint8');
image(uint8(inpaintedImg));



%------------------------------------------------------------------------------
% Load miages and use fillColor as the mask to recognize the target region
%------------------------------------------------------------------------------
function [img,fillImg,fillRegion] = loadimgs(origFileName,fillReFileN,fillColor)
img = imread(origFileName); fillImg = imread(fillReFileN);
fillRegion = fillImg(:,:,1)==fillColor(1) & fillImg(:,:,2)==fillColor(2) & fillImg(:,:,3)==fillColor(3);


%---------------------------------------------------------------------
% Returns the indices for a 9x9 patch centered at pixel p.
%---------------------------------------------------------------------
function [Hp,rows,cols] = getpatch(sz,p,w)
% [x,y] = ind2sub(sz,p);  % 2*w+1 == the patch size
%w=3;  %size is 7x7 default
p=p-1; y=floor(p/sz(1))+1; p=rem(p,sz(1)); x=floor(p)+1;
rows = max(x-w,1):min(x+w,sz(1));
cols = (max(y-w,1):min(y+w,sz(2)))';
Hp = sub2ndx(rows,cols,sz(1));

%----------------------------------------------------------------------
% To define the neighbourhood window
%----------------------------------------------------------------------
function [Np,Rows,Cols] = getNei(sz,p,N)
% 2*N +1 == neighbourhood window size
%N = 25;   % default
p=p-1; Y=floor(p/sz(1))+1; p=rem(p,sz(1)); X=floor(p)+1;
Rows = max(X-N,1):min(X+N,sz(1));
Cols = (max(Y-N,1):min(Y+N,sz(2)))';
Np = sub2ndx(Rows,Cols,sz(1));

%----------------------------------------------------------------------
% To calculate patch sparsity
% Return the patch sparsity and upper limit of the interval and similarity
% between p and pj for later patch inpainting 
%----------------------------------------------------------------------
function [spar,upper,wqj,fi] = sparsity(tarP, windowN,fillRegion,sourceRegion,sz,imglabG,imglab,wi)
sig = 5;
qj = windowN(~(fillRegion(windowN)));
%qqqqqqqq=size(qj)
for i = 1:numel(qj)        
    Hj = getpatch(sz,qj(i),wi);
    check = fillRegion(Hj);
    if any(check(:))
        qj(i) = 0;
    end
end
qj(qj==0) = [];
%sdssfsafdff=size(qj)% 得到在neighbourhood window里且patch都在fillregion里的点
unknown = fillRegion(tarP);
known = logical(~unknown);
untP = tarP(~known);
%wwwwwww=size(tarP)
in=0; wqj = zeros(numel(qj),1);
tarP = tarP(known);
%aaaaaaaa=size(tarP)
fi = zeros(3*numel(untP),1); % 把所有的window里的patch的unknownregion整合成一个一维的列向量
addIn = ones(numel(untP),1);  % 作为在后面的index使用
addIn = addIn.*sz(1)*sz(2);

for kkk = 1:length(qj)
    k = qj(kkk);
    patcherr = 0;
    
    Hjj = getpatch(sz,k,wi); % 对patch的选取有问题。需要修改，目测是因为getpatch这个function，确保当得不到patch时舍去该点
%     %checks = size(Hjj):  % 重新得到每个center对应的patch
    if (numel(Hjj)< ((2*wi+1)^2))
        qj(kkk)=0;
        continue;
    end        %需要考虑到continue对wqj的影响，wqj的值不能为零
    %checkHjj = size(Hjj)
    Hj = Hjj(known);  % known pixel in the patch 
    Hi = Hjj(~known);  % known pixels in the patch in column vector
    %ff=numel(Hj)+numel(Hi)
   seef = imglabG;
    for kk = 1:numel(Hj)
        err = imglabG(tarP(kk)) -imglabG(Hj(kk));
       see= imglabG(tarP(kk));
        patcherr = patcherr + (err^2);
%         err = imglab(tarP(kk)+sz(1)*sz(2))-imglab(Hj(kk)+sz(1)*sz(2));
%         patcherr = patcherr + (err^2);
%         err = imglab(tarP(kk)+2*sz(1)*sz(2))-imglab(Hj(kk)+2*sz(1)*sz(2));
%         patcherr = patcherr + (err^2);
    end
   % dfdfdfdfdf=patcherr;
    ssss = double(exp(-(patcherr/numel(Hj))/(sig^2)));  %计算出每一个还未正则化的wqj
    wqj(kkk)=ssss;
    %sd=size(addIn)
    fi(1:numel(Hi)) = fi(1:numel(Hi)) + imglab(Hi)*wqj(kkk);
    fi((numel(Hi)+1):2*numel(Hi)) = fi((numel(Hi)+1):2*numel(Hi)) + imglab(Hi+addIn)*wqj(kkk);
    fi((2*numel(Hi)+1):3*numel(Hi)) = fi((2*numel(Hi)+1):3*numel(Hi)) + imglab(Hi+addIn.*2)*wqj(kkk);
end

ghghgh=sum(wqj);
if ghghgh==0
    wqj = 0;
    spar=0.200;
    upper=0;
    return;
else
    fi = fi./sum(wqj);
    wqj = wqj./sum(wqj);
end
 
% 正则化wqj  %----------------存在都是NAN的情况，目前已修正估计是之前使用M的时候每个m都不更新  
%cl=sum(wqj)
spar = sqrt((wqj'*wqj)* (numel(qj)/ numel(windowN)));
upper = sqrt(numel(qj)/ numel(windowN));

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

%function ws = selfAdj(
    