use warnings;
use strict;
open IN,"users.dat0";
open OUT,">users.out0";

while(<IN>)
{

	my @line = split /::/,$_;
	if ($line[1] eq 'M')
	{
		$line[1] = 1;
	}
	else
	{
		$line[1] = 0;
	}
	my $out_line = join "::",@line;
	print OUT $out_line;
}

close IN;
close OUT;