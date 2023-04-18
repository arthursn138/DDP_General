%% Differential Dynamic Programming

% Arthur Nascimento - CORE lab @ Georgia Tech
% Hassan Almubarak - ACDS Lab @ Georgia Tech
% nascimento, halmubarak [@gatech.edu]
% Last Update March/04/2023

% Instructions
% 1. call the system's dynamics
% 2. Define DDP and optimization paramters and run vanilla ddp

% Clear workspace and close figures
clear; clc
close all

%% add paths
parentDirectory = fileparts(cd);
addpath(genpath(parentDirectory));

%% Initialize simulation parameters and system dynamics
dt = 0.02; % Discretization
N = 500; % horizon
T_total = dt*N;
T = 0:dt:dt*N-1*dt;

% Dynamics
% % [f, fx, fu, x, u] = quadrotor_dynamics(dt,1);
[f, fx, fu, x, u] = core_cf_linear_dyn(dt);
f_dyn.f = f; f_dyn.fx = fx; f_dyn.fu = fu;
n = length(x);
m = length(u);

% Initial and desired final states -- Watch out for the dynamics model!!!
x0 = zeros(n,1);
xf = zeros(n,1);

% % xf(10) = 10; % X position quadrotor_dynamics
% % xf(11) = -5; % Y position quadrotor_dynamics
% % xf(12) = 15; % Z position quadrotor_dynamics

xf(1) = 2; % X position CF sysID
xf(2) = 5; % Y position CF sysID
xf(3) = 15; % Z position CF sysID

if length(xf)~=n || length(x0)~=n
    error('wrong dimention of x0 and/or xf conditions');
end

sys_par = struct('dt', dt, 'N', N, 'x0', x0, 'xf', xf, 'n', n, 'm', m);

%% Quadratic costs (running cost and terminal cost)
Q = 1*eye(n);
R = 0.5e-3*eye(m);
S = 100*eye(n);

run_cost = @(x, u, deriv_bool) run_quad_cost(x, u, Q, R, xf, deriv_bool);
term_cost = @(x, deriv_bool) terminal_quad_cost(x, xf, S, deriv_bool);

%% Nominal input and state (for faster convergence)
ubar = 0*ones(m, N-1); % nominal control
% ubar(1,:) = 9.81*ones(1, N-1);

xbar=[]; xbar(:,1) = x0;
for k = 1:N-1
    xbar(:, k+1) = f_dyn.f(xbar(:, k), ubar(:, k)); % nominal state
end

%% Optimization parameters
iter= 500;               % max iterations
toler = 1e-3;            % cost change 1e-3

lambda = 1;              % initial value for lambda for regularization
dlambda = 1;             % initial value for dlambda for regularization
lambdaFactor = 1.6;      % lambda scaling factor
lambdaMax = 1e10;        % lambda maximum value
lambdaMin = 1e-6;        % below this value lambda = 0
opt_par = struct('iter', iter, 'toler', toler, 'lambda', lambda,...
    'dlambda', dlambda,'lambdaFactor', lambdaFactor, 'lambdaMax',...
    lambdaMax, 'lambdaMin', lambdaMin);

%% Discrete DDP
[X, U, J, ~, ~, ~, ~, ~, ii, iter_succ, L] = disc_ddp_alg(0,...
    f_dyn, run_cost, term_cost, sys_par, ubar, xbar, opt_par);

%% Plots

% % Declaration of physical significance of the states - changes with model

% % % From quadrotor_dynamics.m
% % X_pos = X(10,:); Y_pos = X(11,:); Z_pos = X(12,:);
% % roll = X(1,:); pitch = X(2,:); yaw = X(3,:);
% % vx = X(7,:); vy = X(8,:); vz = X(9,:);

% From CF linear sysID
angles = zeros(length(X(4,:)));
for i=1:length(X(4,:))
    angles(i) = quat2eul(X(4,i), X(5,i), X(6,i), X(7,i));
end
X_pos = X(1,:); Y_pos = X(2,:); Z_pos = X(3,:);
roll = angles(1,:); pitch = angles(2,:); yaw = angles(3,:);
vx = X(8,:); vy = X(9,:); vz = X(10,:);


% % 3D position
figure()
plot3(X_pos, Y_pos, Z_pos, '-','Color','#D95319','LineWidth',1.5); 
hold on; grid on;
plot3(X_pos(1), Y_pos(1), Z_pos(1), '*b','LineWidth',1); 
plot3(X_pos(end), Y_pos(end), Z_pos(end), 'ok','LineWidth',1.5); 
xlabel('$x$','FontName','Times New Roman','Interpreter','latex');
ylabel('$y$','FontName','Times New Roman','Interpreter','latex');
zlabel('$z$','FontName','Times New Roman','Interpreter','latex');


% % Angles and velocities
figure()
subplot(3,2,1)
plot(T, roll)
title('Roll')
grid on

subplot(3,2,3)
plot(T, pitch)
title('Pitch')
grid on

subplot(3,2,5)
plot(T, yaw)
title('Yaw')
grid on

subplot(3,2,2)
plot(T, vx)
title('V_x')
grid on

subplot(3,2,4)
plot(T, vy)
title('V_y')
grid on

subplot(3,2,6)
plot(T, vz)
title('V_z')
grid on