%
% Simulation of the Fig.7 from 
% I.N. Meshkov. "Electron Cooling: Status and Perspectives".
% Phys. Part. Nucl., Vol. 25 (6) (1994) 631-661 (In Russian:
% Fiz.Elem.Chastits and Atom. Yadra, Vol. 25 (1994) 1487-1560)
%
% The followng formulas are used here:
% (1.36) for perpendicular and longitudinal components
% of the friction force;
% (1.37) for different expressions of Coulomb logatithm;
% (1.35) for number of collisions between electron and ion during 
% their "fast", "adiabatical" and "magnetised" interaction.
%

%
% Fundamental parameters:
%
m_e=9.10938356e-28               % g
m_p=1.672621898e-24              % g
c_light=2.99792458e10            % cm/sec
q_e=4.893204673e-10              % sqrt(g*cm^3/sec^2)
pi=3.14159265358
eVtoErg=1.6021766208e-12         % 1 eV = 1.6...e-12 g*cm^2/sec^2

%
% Parameters of the electron beam (Meshkov, Tabl.III):
%
Ekin=3.e4                        % kinetic energy, eV
omega_p=1.e9                     % plasma frequency, 1/sec
transvT=0.5                      % transversal temperature, eV
longT=2.e-4                      % longitudinal temperature, eV
Bmagn=1.e3*[6 0.6 0.1]'          % magnetic fieldm Gs
aBeam=0.5                        % beam radius, cm
currntBeam=0.5                   % current density, A/cm^2
angSpread=3.                     % angular spread, mrad

%
% Calculated parameters of the electron beam:
%
V0=sqrt(2.*Ekin*eVtoErg/m_e)     % average velocity, cm/sec
vRMStran=sqrt(2.*transvT*eVtoErg/m_e)  % RMS transversal velocity, cm/sec
vRMSlong=sqrt(2.*longT*eVtoErg/m_e)    % RMS longitudinal velocity, cm/sec
