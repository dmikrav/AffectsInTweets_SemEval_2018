
# standard Python modules
import os
import sys
import time
import string
import struct
import msvcrt

# Win32 Extension modules
import win32api
import win32con
import win32ras
import winerror

# DLL pseudo-modules, for directly callng raw Win32 APIs that aren't wrapped
import windll
user32 = windll.module( 'user32' )
rasapi32 = windll.module( 'rasapi32' )

# ----------------------------------------------------------------------------

# name of phonebook entry to dial, in default system phonebook
PHONEBOOK_ENTRY = 'Office VPN'

# ----------------------------------------------------------------------------

# other constants...
RESERVED = 0

# <ras.h>
RASCS_PAUSED = 0x1000
RASCS_DONE   = 0x2000

RASCS_OpenPort = 0
RASCS_PortOpened = 1
RASCS_ConnectDevice = 2
RASCS_DeviceConnected = 3
RASCS_AllDevicesConnected = 4
RASCS_Authenticate = 5
RASCS_AuthNotify = 6 # sending username and password...
RASCS_AuthRetry = 7
RASCS_AuthCallback = 8
RASCS_AuthChangePassword = 9
RASCS_AuthProject = 10
RASCS_AuthLinkSpeed = 11
RASCS_AuthAck = 12
RASCS_ReAuthenticate = 13
RASCS_Authenticated = 14
RASCS_PrepareForCallback = 15
RASCS_WaitForModemReset = 16
RASCS_WaitForCallback = 17
RASCS_Projected = 18

#if (WINVER >= 0x400)
RASCS_StartAuthentication = 19
RASCS_CallbackComplete = 20
RASCS_LogonNetwork = 21
#endif

RASCS_SubEntryConnected = 22
RASCS_SubEntryDisconnected = 23

RASCS_Interactive = RASCS_PAUSED
RASCS_RetryAuthentication = RASCS_PAUSED + 1
RASCS_CallbackSetByCaller = RASCS_PAUSED + 2
RASCS_PasswordExpired = RASCS_PAUSED + 3

RASCS_Connected = RASCS_DONE
RASCS_Disconnected = RASCS_DONE + 1

RASP_PppIp=0x8021

# ----------------------------------------------------------------------------

# return string formatted like default exception traceback
def format_traceback():
    import sys
    x = sys.exc_info()
    buf = 'Traceback (innermost last):\n'
    import traceback
    tblist = traceback.extract_tb( x[2] )
    for tb in tblist:
        buf = buf + '  File "%s", line %d, in %s\n    %s\n' % tb
    buf = buf + '%s: %s\n' % ( x[0].__name__, str( x[1] ) )
    return buf

# print traceback to stdout
def print_traceback( buf=None ):
    if not buf:
        buf = format_traceback()
    import sys
    sys.stderr.write( buf )

# ----------------------------------------------------------------------------

# get HWND of window matching given class name and title
def FindWindow( wndclassname, windowname ):
    hwnd = user32.FindWindow( windll.cstring( wndclassname ), windll.cstring( windowname ) )
    if hwnd == 0:
        errcode = win32api.GetLastError()
        errmsg = win32api.FormatMessage( errcode )
        raise win32api.error( errcode, 'FindWindow', errmsg )
    return hwnd

# ----------------------------------------------------------------------------

# dial a phonebook entry if it isn't already connected
# run the GUI tool so it looks nicer
# wait for the dialog to close
# check connection status afterwards
def DialPhoneBookEntry( phonebook_entry ):
    isconnected = 0
    conns = win32ras.EnumConnections()
    for conn in conns:
        #print conn
        if conn[1] == phonebook_entry:
            isconnected = 1

    if isconnected:
        print 'Connected to', phonebook_entry
    else:
        print 'Dialing %s . . .' % phonebook_entry
        win32api.WinExec( 'rasphone -d \"%s\"' % phonebook_entry )
        # TODO: handle Cancel within rasphone
        status = RASCS_Disconnected
        while not isconnected:
            win32api.Sleep( 1000 )
            conns = win32ras.EnumConnections()
            for conn in conns:
                if conn[1] == phonebook_entry:
                    hConn = conn[0]
                    status = win32ras.GetConnectStatus( hConn )
                    # intermediate states 5 = RASCS_Authenticate, 14=RASCS_Authenticated
                    if status[0] == RASCS_Authenticate:
                        if status != status[0]:
                            status = status[0]
                            print 'Authenticating...'
                    elif status[0] == RASCS_Authenticated:
                        if status != status[0]:
                            status = status[0]
                            print 'Authenticated.'
                    elif status[0] == RASCS_Connected:
                        print 'Connected.'
                        isconnected = 1
                        break
                    else:
                        print 'status:', status
            else:
                # *** this only works in NT4
                # *** need to figure out equiv for W2K
                winver = win32api.LOWORD( win32api.GetVersion() )
                if winver < 5:
                    try:
                        hwnd = FindWindow( '#32770', 'Connecting to %s...' % phonebook_entry )
                    except win32api.error, err:
                        if err[0] == winerror.ERROR_PROC_NOT_FOUND:
                            print 'Connection cancelled.'
                            time.sleep( 1 )
                            return
        #while not connected
    #else not connected

    return isconnected
#DialPhoneBookEntry

# ----------------------------------------------------------------------------

# return dotted decimal IP address for RAS connection
def GetRasIpAddress( phonebook_entry ):
    connlist = win32ras.EnumConnections()
    #[(655360, 'Office PPTP', 'VPN', 'RASPPTPM')]
    for conninfo in connlist:
        if conninfo[1] == phonebook_entry:
            break
    else:
        print "RAS connection '%s' not found" % phonebook_entry
        sys.exit(1)

    hrasconn = conninfo[0]
    structsize = struct.calcsize( 'L L 16s' )

    dw = windll.membuf( struct.calcsize( 'i' ) )
    dw.mb.write( struct.pack( 'i', structsize ) )

    buf = windll.membuf( structsize )
    buf.mb.write( struct.pack( 'L', structsize ) ) # MUST set struct member

    # on W2K/laptop, need a bit of a delay???
    maxretries = 10
    count = 0
    while count < maxretries:
        err = rasapi32.RasGetProjectionInfo( hrasconn, RASP_PppIp, buf, dw )
        if err == 0:
            break
        else:
            count = count + 1
            if count < maxretries:
                win32api.Sleep( 1000 )
            else:
                print 'RasGetProjectionInfo error %d' % err
                print 'go look up the error code in <ras.h>'
                sys.exit( 1 )

    dwSize, err, rawipaddr = struct.unpack( 'L L 16s', buf.mb.read() )
    ipaddr, junk = string.split( rawipaddr,'\0', 1 )
    #'192.168.91.4\000_ma'

    print 'IP address:', ipaddr
    return ipaddr
#GetRasIpAddress

# ----------------------------------------------------------------------------

# update routing table for office subnets given IP address to use as gateway
def SetupOfficeRouting( gateway ):
    os.system( 'route add 192.197.216.0 mask 255.255.252.0 %s' % gateway )
    os.system( 'route add 192.168.0.0 mask 255.255.128.0 %s' % gateway )
    os.system( 'route add 204.174.12.32 mask 255.255.255.240 %s' % gateway )
    os.system( 'route add 204.174.12.48 mask 255.255.255.240 %s' % gateway )
    os.system( 'route add 204.174.12.112 mask 255.255.255.240 %s' % gateway )
    print 'IP routing established for office network using PPTP connection.'

# ----------------------------------------------------------------------------

# hangup RAS connection, if connected
def HangUpConnection( phonebook_entry ):
    connlist = win32ras.EnumConnections()
    #[(655360, 'Office PPTP', 'VPN', 'RASPPTPM')]
    for conninfo in connlist:
        if conninfo[1] == phonebook_entry:
            hrasconn = conninfo[0]
            win32ras.HangUp( hrasconn )
            break
    else:
        print 'Not connected.'
        #print "RAS connection '%s' not found" % phonebook_entry
        sys.exit(1)

# ----------------------------------------------------------------------------

# manage Office PPTP connection
def vpnadmin():
    USAGE = 'Usage: vpnadmin { dial | hangup | print }'

    argc = len( sys.argv )
    if argc != 2:
        print USAGE
        sys.exit( 1 )

    cmd = sys.argv[1]
    if cmd == 'dial':
        try:
            isp_entry = os.environ['ISP_PHONEBOOK_ENTRY']
            DialPhoneBookEntry( isp_entry )
            pass
        except KeyError:
            pass
        DialPhoneBookEntry( PHONEBOOK_ENTRY )
        ipaddr = GetRasIpAddress( PHONEBOOK_ENTRY )
        SetupOfficeRouting( ipaddr )
    elif cmd == 'hangup':
        HangUpConnection( PHONEBOOK_ENTRY )
    elif cmd == 'print':
        os.system( 'ipconfig /all' )
        os.system( 'route print' )
    else:
        print 'Unrecognized command:', cmd
        print USAGE
        sys.exit( 1 )

def main():
    try:
        vpnadmin()
    except SystemExit:
        pass
    except:
        print_traceback()
        raw_input( 'Press ENTER to finish . . . ' )

if __name__ == '__main__':
    main()
Previous message (by thread): Python Script to dial a VPN connection, automate VPN routing table updates.
Next message (by thread): Python Script to dial a VPN connection, automate VPN routing table updates.
Messages sorted by: [ date ] [ thread ] [ subject ] [ author ]
More information about the Python-list mailing list