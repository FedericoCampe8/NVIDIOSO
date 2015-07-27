#!/usr/bin/perl -w
use strict;
use warnings;
my $user = $ENV{'USER'};
my $from = 'mail@invidioso.com';
my $to = 'fede.campe@gmail.com';
my $sendmail = 'sendEmail-v1.56/sendEmail.pl';
my $datestring;
my $logmessage = "iNVIDIOSO: Attached log files. Please check urgently!\n\n";
my $processFileHandler;
my $logFileHandler;
my $emailFlag=0;
my @logFiles = `ls | grep log`;

$datestring = gmtime();
$emailFlag = 1;

my $num_args;
my $arg;

# (1) quit unless we have the correct number of command-line args
$num_args= $#ARGV + 1;
if($num_args > 1)
{
    print "Usage: SendLog.pl [-h | local_mailbox\@domain_name]\n";
    exit;
}

$arg=$ARGV[0];
if($num_args == 1)
{
    $arg=$ARGV[0];
    if ( $arg eq "-h" )
    {
        print "Usage: SendLog.pl [-h | local_mailbox\@domain_name]\n";
        exit;
    }
    else
    {
        $to = $arg
    }
}

if($emailFlag)
{
    foreach my $file (@logFiles) 
    {
        system "${sendmail} -o tls=auto -m \"${logmessage}\" -u \"CRITICAL: ${user}: Logs sent on GMT ${datestring} \" -t ${to} -f ${from} -a $file";
    }
}
