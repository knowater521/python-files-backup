use strict;
use warnings;

my $infile = $ARGV[0];
open OUT, ">tmp.txt";
open IN ,$infile;


while(<IN>)
{
    print OUT;
}

close IN;
close OUT;

open OUT,">$infile";
open IN,"tmp.txt";
while(<IN>)
{
    if(/\t/)
    {
        s/\t/    /g;
        print OUT;
        
    }
    else
    {
        print OUT;
    }
}

close IN;
close OUT;
system("del tmp.txt");