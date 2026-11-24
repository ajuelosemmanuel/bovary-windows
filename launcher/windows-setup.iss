#define KitName "bovary-0.2.0a18-windows-x64-setup"
[Setup]
AppId=BovaryMiniWindows
AppName=Bovary Mini
AppVersion=0.2.0a18
AppPublisher=Bovary contributors
AppPublisherURL=https://github.com/ajuelosemmanuel/bovary-windows
DefaultDirName={localappdata}\Programs\Bovary Mini
DefaultGroupName=Bovary
DisableProgramGroupPage=yes
DisableDirPage=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
WizardStyle=modern
OutputBaseFilename={#KitName}
OutputDir=installer-output
Compression=zip
SolidCompression=no
UninstallDisplayName=Bovary Mini
SetupLogging=yes

[Files]
Source: "verified-kit\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{userprograms}\Bovary Mini"; Filename: "{app}\Start Bovary.cmd"; WorkingDir: "{app}"

[Run]
Filename: "{app}\Start Bovary.cmd"; WorkingDir: "{app}"; Description: "Open Bovary and choose a model"; Flags: shellexec postinstall nowait skipifsilent
