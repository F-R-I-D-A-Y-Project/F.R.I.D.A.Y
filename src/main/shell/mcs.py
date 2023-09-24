import subprocess, os, sys, pexpect, platform
class MCS:
    '''
        This class is responsible for the communication between the chatbot and the shell
    '''

    def __init__(self, *, init_cmd=None, prompt=None, command=None) -> None:
        self.platform = platform.system()
        # if 
        if self.platform == 'Windows':
            self.init_cmd = "cmd.exe"
        elif self.platform == 'Linux' or self.platform == 'Darwin':
            self.init_cmd = "bash"
        self.command, self.prompt = command, prompt
        pass

    def send(self, message: str) -> None:
        '''
            This method sends a message to the shell
        '''
        stdout, stderr = self.popen.communicate(bytes(message, 'utf-8'))
        return stdout
    def __enter__(self) -> pexpect.spawn:
        self.child_process = pexpect.spawn(self.init_cmd)
        if self.command != None:
            self.child_process.sendline(self.command)
        return self.child_process
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.child_process.close()

    pass

def main():
    with MCS(command="echo 'Hello, World!'") as mcs:
        mcs.send("")

if __name__ == "__main__":
    main()