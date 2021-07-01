%%
clear;
close all;
clc;

%% Schwefel function do sprawdzenia
syms x1;
syms x2;

learn_rate = 0.01;
acceleration = 0.9;
momentum = 0.9;

epsilon = 0.00001;
max_iter = 500;

%f = @(x1, x2) x1.^2 + x2.^2; % Paraboloid 2,2
f = @(x1, x2) x1.^2 - x2.^2; % Saddle point 2,0
%f = @(x1, x2) (x1.^2 + x2 - 11).^2 + (x1 + x2.^2 - 7).^2; % Himmelblau 0,0
%f = @(x1, x2) sin(x2) .* exp((1 - cos(x1)).^2) + cos(x1) .* exp((1 - sin(x2)).^2) + (x1 - x2).^2; % Mishra 3,-5;
%f = @(x1, x2) (1 - x1).^2 + 100 .* (x2 - x1.^2).^2; %Rosenbrock 0,0

[X, Y] = meshgrid(-10:.5:10, -10:.5:10);
%Z = X.^2 + Y.^2; % Paraboloid
Z = X.^2 - Y.^2; % Saddle point
%Z = (X.^2 + Y - 11).^2 + (X + Y.^2 - 7).^2; % Himmelblau
%Z = sin(Y) .* exp((1 - cos(X)).^2) + cos(X) .* exp((1 - sin(Y)).^2) + (X - Y).^2; % Mishra
%Z = (1 - X).^2 + 100 .* (Y - X.^2).^2; % Rosenbrock

x0 = [2,0];
x_sgd = SGD(x0, f, x1, x2, learn_rate, epsilon, max_iter); 
x_mom = SGD_momentum(x0, f, x1, x2, learn_rate, momentum, epsilon, max_iter);
x_nag = NAG(x0, f, x1, x2, learn_rate, acceleration, epsilon, max_iter);
x_adagrad = ADAGRAD(x0, f, x1, x2, learn_rate, epsilon, max_iter);
x_rms = RMSPROP(x0, f, x1, x2, learn_rate, epsilon, max_iter);
x_adadelta = ADADELTA(x0, f, x1, x2, epsilon, max_iter);
x_adam = ADAM(x0, f, x1, x2, learn_rate, epsilon, max_iter);
x_adamax = ADAMAX(x0, f, x1, x2, learn_rate, epsilon, max_iter);
x_nadam = NADAM(x0, f, x1, x2, learn_rate, epsilon, max_iter);

x_sgd = [x0; x_sgd];
x_mom = [x0; x_mom];
x_nag = [x0; x_nag];
x_adagrad = [x0; x_adagrad];
x_rms = [x0; x_rms];
x_adadelta = [x0; x_adadelta];
x_adam = [x0; x_adam];
x_adamax = [x0; x_adamax];
x_nadam = [x0; x_nadam];

f_sgd = f(x_sgd(:,1), x_sgd(:,2));
f_mom = f(x_mom(:,1), x_mom(:,2));
f_nag = f(x_nag(:,1), x_nag(:,2));
f_adagrad = f(x_adagrad(:,1), x_adagrad(:,2));
f_rms = f(x_rms(:,1), x_rms(:,2));
f_adadelta = f(x_adadelta(:,1), x_adadelta(:,2));
f_adam = f(x_adam(:,1), x_adam(:,2));
f_adamax = f(x_adamax(:,1), x_adamax(:,2));
f_nadam = f(x_nadam(:,1), x_nadam(:,2));

%% Wykres zbieżności
figure(1);
plot(f_sgd);
hold on;
plot(f_mom);
hold on;
plot(f_nag);
hold on;
plot(f_adagrad);
hold on;
plot(f_rms);
hold on;
plot(f_adadelta);
hold on;
plot(f_adam);
hold on;
plot(f_adamax);
hold on;
plot(f_nadam);
title("Porównanie zbieżności algorytmów optymalizacji");
legend('sgd','mom','nag','adg','rms','add','adam','adamax','nadam');

%% Wykres 2d
subplot(3,3,1);
fcontour(f);
hold on; 
plot(x_sgd(:,1), x_sgd(:,2),'-o');
title("SGD x* = " + x_sgd(end,:));

subplot(3,3,2);
fcontour(f);
hold on;
plot(x_mom(:,1), x_mom(:,2),'-o');
title("SGD with momentum x* = " + x_mom(end,:));

subplot(3,3,3);
fcontour(f);
hold on;
plot(x_nag(:,1), x_nag(:,2),'-o');
title("Nesterov accelerated x* = " + x_nag(end,:));

subplot(3,3,4);
fcontour(f);
hold on;
plot(x_adagrad(:,1), x_adagrad(:,2),'-o');
title("Adagrad x* = " + x_adagrad(end,:));

subplot(3,3,5);
fcontour(f);
hold on;
plot(x_adadelta(:,1), x_adadelta(:,2),'-o');
title("Adadelta x* = " + x_adadelta(end,:));

subplot(3,3,6);
fcontour(f);
hold on;
plot(x_rms(:,1), x_rms(:,2),'-o');
title("RMSprop x* = " + x_rms(end,:));

subplot(3,3,7);
fcontour(f);
hold on;
plot(x_adam(:,1), x_adam(:,2), '-o');
title("Adam x* = " + x_adam(end,:));

subplot(3,3,8);
fcontour(f);
hold on;
plot(x_adamax(:,1), x_adamax(:,2), '-o');
title("Adamax x* = " + x_adamax(end,:));

subplot(3,3,9);
fcontour(f);
hold on;
plot(x_nadam(:,1), x_nadam(:,2), '-o');
title("Nadam x* = " + x_nadam(end,:));

%% Wykres 3d
subplot(3,3,1);
mesh(X, Y, Z, 'FaceAlpha','0.5');
hold on;
scatter3(x_sgd(:,1), x_sgd(:,2), f(x_sgd(:,1), x_sgd(:,2)),'filled','r');
title('SGD');

subplot(3,3,2);
mesh(X, Y, Z, 'FaceAlpha','0.5');
hold on;
scatter3(x_mom(:,1), x_mom(:,2), f(x_mom(:,1), x_mom(:,2)),'filled','r');
title('SGD with momentum');

subplot(3,3,3);
mesh(X, Y, Z, 'FaceAlpha','0.5');
hold on;
scatter3(x_nag(:,1), x_nag(:,2), f(x_nag(:,1), x_nag(:,2)),'filled','r');
title('Nesterov accelerated');

subplot(3,3,4);
mesh(X, Y, Z, 'FaceAlpha','0.5');
hold on;
scatter3(x_adagrad(:,1), x_adagrad(:,2), f(x_adagrad(:,1), x_adagrad(:,2)),'filled','r');
title('Adagrad');

subplot(3,3,5);
mesh(X, Y, Z, 'FaceAlpha','0.5');
hold on;
scatter3(x_adadelta(:,1), x_adadelta(:,2), f(x_adadelta(:,1), x_adadelta(:,2)),'filled','r');
title('Adadelta');

subplot(3,3,6);
mesh(X, Y, Z, 'FaceAlpha','0.5');
hold on;
scatter3(x_rms(:,1), x_rms(:,2), f(x_rms(:,1), x_rms(:,2)),'filled','r');
title('RMSprop');

subplot(3,3,7);
mesh(X, Y, Z, 'FaceAlpha','0.5');
hold on;
scatter3(x_adam(:,1), x_adam(:,2), f(x_adam(:,1), x_adam(:,2)),'filled','r');
title('Adam');

subplot(3,3,8);
mesh(X, Y, Z, 'FaceAlpha','0.5');
hold on;
scatter3(x_adamax(:,1), x_adamax(:,2), f(x_adamax(:,1), x_adamax(:,2)),'filled','r');
title('ADAMAX');

subplot(3,3,9);
mesh(X, Y, Z, 'FaceAlpha','0.5');
hold on;
scatter3(x_nadam(:,1), x_nadam(:,2), f(x_nadam(:,1), x_nadam(:,2)),'filled','r');
title('Nadam');

%% Wykres zbieżności
subplot(3,3,1);
bar3(x_sgd);
title("SGD - iter = " + length(x_sgd));

subplot(3,3,2);
bar3(x_adagrad);
title("Adagrad - iter = " + length(x_adagrad));

subplot(3,3,3);
bar3(x_mom);
title("SGD with momentum - iter = " + length(x_mom));

subplot(3,3,4);
bar3(x_nag);
title("Nesterov accelerated - iter = " + length(x_nag));

subplot(3,3,5);
bar3(x_rms);
title("RMSprop - iter = " + length(x_rms));

subplot(3,3,6);
bar3(x_adadelta);
title("Adadelta - iter = " + length(x_adadelta));

subplot(3,3,7);
bar3(x_adam);
title("Adam - iter = " + length(x_adam));

subplot(3,3,8);
bar3(x_adamax);
title("Adamax - iter = " + length(x_adamax));

subplot(3,3,9);
bar3(x_nadam);
title("Nadam - iter = " + length(x_nadam));

%% Funkcje

function x = NADAM(x0, f, x1, x2, alfa, eps, max_iter)
   go = true;
   x = [];
   iter = 0;
   v = [0, 0];
   m = [0, 0];
   while(go)
       iter = iter + 1;
        [x_, m, v] = Update_NADAM(x0, f, x1, x2, alfa, m, v, iter);
        x = [x; x_];
        if(norm(x(end,:) - x0) <= eps)
           break;
        end
        if(iter == max_iter)
           break; 
        end
        x0 = x(end,:);
   end
end

function [x, m, v] = Update_NADAM(x0, f, x1, x2, alfa, m_bef, v_bef, iter)
    x = zeros(1, length(x0));
    m_est = zeros(1, length(x0));
    v_est = zeros(1, length(x0));
    grad = Eval_gradient(x0, f, x1, x2);
    b1 = 0.9;
    b2 = 0.999;
    
    m(1) = b1 * m_bef(1) + (1 - b1) * grad(1);
    m(2) = b1 * m_bef(2) + (1 - b1) * grad(2);
    
    v(1) = b2 * v_bef(1) + (1 - b2) * grad(1)^2;
    v(2) = b2 * v_bef(2) + (1 - b2) * grad(2)^2;
    
    m_est(1) = m(1)/(1 - b1^iter);
    m_est(2) = m(2)/(1 - b1^iter);
    
    v_est(1) = v(1)/(1 - b2^iter);
    v_est(2) = v(2)/(1 - b2^iter);
    
    x(1) = x0(1) - (alfa/sqrt(v_est(1)) + 0.000000001) * (b1 * m_est(1) + ((1 - b1)*grad(1)/(1 - b1^iter)));
    x(2) = x0(2) - (alfa/sqrt(v_est(2)) + 0.000000001) * (b1 * m_est(2) + ((1 - b1)*grad(2)/(1 - b1^iter)));
end

function x = ADAMAX(x0, f, x1, x2, alfa, eps, max_iter)
   go = true;
   x = [];
   iter = 0;
   v = [0, 0];
   m = [0, 0];
   while(go)
       iter = iter + 1;
        [x_, m, v] = Update_ADAMAX(x0, f, x1, x2, alfa, m, v, iter);
        x = [x; x_];
        if(norm(x(end,:) - x0) <= eps)
           break;
        end
        if(iter == max_iter)
           break; 
        end
        x0 = x(end,:);
   end    
end

function [x, m, v] = Update_ADAMAX(x0, f, x1, x2, alfa, m_bef, v_bef, iter)
    x = zeros(1, length(x0));
    m_est = zeros(1, length(x0));
    v_est = zeros(1, length(x0));
    grad = Eval_gradient(x0, f, x1, x2);
    b1 = 0.9;
    b2 = 0.999;
    
    m(1) = b1 * m_bef(1) + (1 - b1) * grad(1);
    m(2) = b1 * m_bef(1) + (1 - b1) * grad(2);
    
    v(1) = max([b2 * v_bef(1), abs(grad(1))]);
    v(2) = max([b2 * v_bef(2), abs(grad(2))]);
    
    x(1) = x0(1) - (alfa/(1 - b1^iter)) * m(1)/v(1);
    x(2) = x0(2) - (alfa/(1 - b1^iter)) * m(2)/v(2);    
end

function x = ADAM(x0, f, x1, x2, alfa, eps, max_iter)
   go = true;
   x = [];
   iter = 0;
   v = [0, 0];
   m = [0, 0];
   while(go)
       iter = iter + 1;
        [x_, m, v] = Update_ADAM(x0, f, x1, x2, alfa, m, v, iter);
        x = [x; x_];
        if(norm(x(end,:) - x0) <= eps)
           break;
        end
        if(iter == max_iter)
           break; 
        end
        x0 = x(end,:);
   end
end

function [x, m, v] = Update_ADAM(x0, f, x1, x2, alfa, m_bef, v_bef, iter)
    x = zeros(1, length(x0));
    m_est = zeros(1, length(x0));
    v_est = zeros(1, length(x0));
    grad = Eval_gradient(x0, f, x1, x2);
    b1 = 0.9;
    b2 = 0.999;
    
    m(1) = b1 * m_bef(1) + (1 - b1) * grad(1);
    m(2) = b1 * m_bef(1) + (1 - b1) * grad(2);
    
    v(1) = b2 * v_bef(1) + (1 - b2) * grad(1)^2;
    v(2) = b2 * v_bef(2) + (1 - b2) * grad(2)^2;
    
    m_est(1) = m(1)/(1 - b1^iter);
    m_est(2) = m(2)/(1 - b1^iter);
    
    v_est(1) = v(1)/(1 - b2^iter);
    v_est(2) = v(2)/(1 - b2^iter);
    
    x(1) = x0(1) - (alfa/sqrt(v_est(1)) + 0.000000001) * m_est(1);
    x(2) = x0(2) - (alfa/sqrt(v_est(2)) + 0.000000001) * m_est(2);
end

function x = ADADELTA(x0, f, x1, x2, eps, max_iter)
    go = true;
    x = [];
    iter = 0;
    Eg = [0.001, 0.001];
    Ex = [0.001, 0.001]; % Testowa inicjalizacja bo w zerze są zbyt wolno zbieżne
    while(go)
        iter = iter + 1;
        [x_, Eg, Ex] = Update_ADADELTA(x0, f, x1, x2, Eg, Ex);
        x = [x; x_];
        if(norm(x(end,:) - x0) <= eps)
           break;
        end
        if(iter == max_iter)
           break; 
        end
        x0 = x(end,:);
    end 
end

function [x, Eg, Ex] = Update_ADADELTA(x0, f, x1, x2, E_g_bef, Ex_bef)
    x = zeros(1, length(x0));
    grad = Eval_gradient(x0, f, x1, x2);
    
    p = 0.95;
    
    Eg(1) = p*E_g_bef(1) + (1 - p) * grad(1)^2;
    Eg(2) = p*E_g_bef(2) + (1 - p) * grad(2)^2;
    
    RMS_x_1 = sqrt(Ex_bef(1) + 0.000000001);
    RMS_x_2 = sqrt(Ex_bef(2) + 0.000000001);
    
    
    RMS_g_1 = sqrt(Eg(1) + 0.000000001);
    RMS_g_2 = sqrt(Eg(2) + 0.000000001);
    
    delta_x1 = RMS_x_1/RMS_g_1* grad(1);
    delta_x2 = RMS_x_2/RMS_g_2* grad(2);
    
    x(1) = x0(1) - delta_x1;
    x(2) = x0(2) - delta_x2;
    
    Ex(1) = p*Ex_bef(1) + (1 - p) * delta_x1^2;
    Ex(2) = p*Ex_bef(2) + (1 - p) * delta_x2^2;
end

function x = RMSPROP(x0, f, x1, x2, alfa, eps, max_iter)
    go = true;
    x = [];
    iter = 0;
    E = zeros(1, length(x0));
     while(go)
        iter = iter + 1;
        [x_, E] = Update_RMSPROP(x0, f, x1, x2, alfa, E);
        x = [x; x_];
        if(norm(x(end,:) - x0) <= eps)
           break;
        end
        if(iter == max_iter)
           break; 
        end
        x0 = x(end,:);
    end
    
end

function [x, E] = Update_RMSPROP(x0, f, x1, x2, alfa, E_bef)
    x = zeros(1, length(x0));
    grad = Eval_gradient(x0, f, x1, x2);
    
    E(1) = 0.9*E_bef(1) + (1 - 0.9) * grad(1)^2;
    E(2) = 0.9*E_bef(1) + (1 - 0.9) * grad(2)^2;
    
    learn1 = alfa/(sqrt(E(1) + 0.000000001));
    learn2 = alfa/(sqrt(E(2) + 0.000000001));
    
    x(1) = x0(1) - learn1 * grad(1);
    x(2) = x0(2) - learn2 * grad(2);
end

function x = ADAGRAD(x0, f, x1, x2, alfa, eps, max_iter)
    go = true;
    x = [];
    iter = 0;
    grad_sum = zeros(1, length(x0));
    while(go)
        iter = iter + 1;
        [x_, grad_sum] = Update_ADAGRAD(x0, f, x1, x2, alfa, grad_sum);
        x = [x; x_];
        if(norm(x(end,:) - x0) <= eps)
           break;
        end
        if(iter == max_iter)
           break; 
        end
        x0 = x(end,:);
    end
end

function [x, grad_sum] = Update_ADAGRAD(x0, f, x1, x2, alfa, grad_bef)
    x = zeros(1, length(x0));
    grad = Eval_gradient(x0, f, x1, x2);
    
    grad_sum(1) = grad_bef(1) + grad(1)^2;
    grad_sum(2) = grad_bef(2) + grad(2)^2;
    
    learn1 = alfa/(sqrt(grad_sum(1) + 0.000000001));
    learn2 = alfa/(sqrt(grad_sum(2) + 0.000000001));
    
    x(1) = x0(1) - learn1 * grad(1);
    x(2) = x0(2) - learn2 * grad(2);
end

function x = NAG(x0, f, x1, x2, alfa, gamma, eps, max_iter)
    go = true;
    x = [];
    iter = 0;
    v = zeros(1, length(x0));
    while(go)
        iter = iter + 1;
        [x_, v] = NAG_update(x0, v, f, x1, x2, alfa, gamma);
        x = [x; x_];
        if(norm(x(end,:) - x0) <= eps)
           break;
        end
        if(iter == max_iter)
           break; 
        end
        x0 = x(end,:);
    end
end

function [x, v] = NAG_update(x0, v_before, f, x1, x2, alfa, gamma)
    x = zeros(1, length(x0));
    v = zeros(1, length(x0));
    grad = Eval_gradient(x0 - gamma*v_before, f, x1, x2);
    v(1) = gamma * v_before(1) + alfa * grad(1);
    v(2) = gamma * v_before(2) + alfa * grad(2);
    x(1) = x0(1) - v(1);
    x(2) = x0(2) - v(2);
end

function x = SGD_momentum(x0, f, x1, x2, alfa, gamma, eps, max_iter)
    go = true;
    x = [];
    iter = 0;
    v = zeros(1, length(x0));
    while(go)
        iter = iter + 1;
        [x_, v] = Momentum_gradient_descent(x0, v, f, x1, x2, alfa, gamma);
        x = [x; x_];
        if(norm(x(end,:) - x0) <= eps)
           break;
        end
        if(iter == max_iter)
           break; 
        end
        x0 = x(end,:);
    end
end

function [x, v] = Momentum_gradient_descent(x0, v_before, f, x1, x2, alfa, gamma)
    x = zeros(1, length(x0));
    v = zeros(1, length(x0));
    grad = Eval_gradient(x0, f, x1, x2);
    v(1) = gamma * v_before(1) + alfa * grad(1);
    v(2) = gamma * v_before(2) + alfa * grad(2);
    x(1) = x0(1) - v(1);
    x(2) = x0(2) - v(2);
end

function x = SGD(x0, f, x1, x2, alfa, eps, max_iter)
    go = true;
    x = [];
    iter = 0;
    while(go)
        iter = iter + 1;
        x = [x; Simple_gradient_descent(x0, f, x1, x2, alfa)];
        if(norm(x(end,:) - x0) <= eps)
           break;
        end
        if(iter == max_iter)
           break; 
        end
        x0 = x(end,:);
    end
end

function x = Simple_gradient_descent(x0, f, x1, x2, alfa)
    x = zeros(1, length(x0));
    grad = Eval_gradient(x0, f, x1, x2);
    x(1) = x0(1) - alfa * grad(1);
    x(2) = x0(2) - alfa * grad(2);
end

function grad = Eval_gradient(x0, f, x1, x2)
    u = 0.001;
    u1 = sqrt(u) * x0(1) + u;
    u2 = sqrt(u) * x0(2) + u;
    dx1 = (f(x0(1) + u1, x0(2)) - f(x0(1), x0(2))) / u1;
    dx2 = (f(x0(1), x0(2) + u2) - f(x0(1), x0(2))) / u2;
    grad = [dx1, dx2];
end
