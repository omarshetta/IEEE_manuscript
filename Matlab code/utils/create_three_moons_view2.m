    function [data]= create_three_moons_view2(N,sigmad,d1,d2)
    
%      rng('default')  % For reproducibility
    % moon 1
    phi1 = rand(N,1)  * pi;
    r1 = 1;
    rb = sigmad * randn(N,1);
    %rb = laprnd(N1, 1, 0, sigmad);
    ab = rand(N,1) * 2 * pi;
    b = rb .*exp(1i.*ab);
    bx = real(b);
    by = imag(b);
    moon1x = cos(phi1) .* r1 + 1 + bx; % 0.5
    moon1y = -sin(phi1) .* r1 + by-(d1-1)/2;
%     figure (10)
%      plot(moon1x , moon1y,'xb');
     
     
      % moon 2
    phi2 = rand(N,1)  * pi;
    r2 = 1;
    rb = sigmad * randn(N,1);
    ab = rand(N,1) * 2 * pi;
    b = rb .*exp(1i.*ab);
    bx = real(b);
    by = imag(b);
%     moon2x = cos(phi2) .* r2 - 0.5 + bx;
%     moon2y = sin(phi2) .* r2 + by+(d-1)/2;
     moon2x = cos(phi2) .* r2 + 0.4 + bx;
     moon2y = sin(phi2) .* r2 + by+(d2-1)/2;
%      hold on;
%      plot(moon2x , moon2y,'xr');    
     
     d=0.6;
    % moon 3
    phi2 = rand(N,1)  * pi;
    r2 = 1;
    rb = sigmad * randn(N,1);
    ab = rand(N,1) * 2 * pi;
    b = rb .*exp(1i.*ab);
    bx = real(b);
    by = imag(b);
%     moon3x = cos(phi2) .* r2 + 0.4 + bx;
%     moon3y = sin(phi2) .* r2 + by+(d-1)/2;
     moon3x = cos(phi2) .* r2 - 0.5 + bx;
     moon3y = sin(phi2) .* r2 + by+(d1-1)/2;
%      hold on;
%      plot(moon3x , moon3y,'xk');
%      axis equal
 
    data = [moon1x,moon1y;moon2x,moon2y;moon3x,moon3y]';
    
    end