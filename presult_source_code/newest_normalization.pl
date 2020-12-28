#!/usr/bin/perl -w
use strict;
use Getopt::Long;

#-----------------------------------------------------------------------------
#----------------------------------- MAIN ------------------------------------
#-----------------------------------------------------------------------------

my $usage = "
Syntax:\nperl normalization.pl --input <input_file> --output <output_file> --missing <character representing missing values> --csv --var <variables> --with_id

Options:
--csv: 	this instructs the script to assume the input file uses comma as column separators.
   	Otherwise, it assumes tab is the separator by default.
--var <variables>: this option instructs the script to output only variables being listed 
	and ignore other variables. It accepts either comma separated variable names or a 
	file containing variable lists (each variable in a separate line).
--with_id: tells the script that the first column is the ID column and should not be normalized.


Note:
1. By default '?' will be used as the missing character.
3. The last column is the outcome variable.
";

my %opt;
my $opt_success = GetOptions(	'output|o=s'	=> \$opt{output},
				'input|i=s'	=> \$opt{input},
				'missing|m=s'	=> \$opt{missing},
				'csv'		=> \$opt{csv},
				'var=s'		=> \$opt{var},
				'with_id'	=> \$opt{with_id},
			);

die $usage unless defined $opt{output} and defined $opt{input};

$opt{missing} = "?" unless defined $opt{missing};

my $want_vars = parse_opt_var($opt{var}) if defined $opt{var};

my @data;
my @ids;
my $num_vars;
my @headers;
open FH, $opt{input} or die "ERROR:Cannot open input file!\n";
while (my $line = <FH>) {
	chomp $line;

	# parse header
	if ($line =~/^#/) {
		$line =~ s/^#+//g;
		if ($opt{csv}) {
			@headers = split /,/, $line;
		} else{	@headers = $line =~ /(\S+)/g; }

		shift @headers if $opt{with_id};
		next;
	}

	# split lines	
	my @i;
	if ($opt{csv}) { @i = split /,/, $line; }
	else {	@i =  $line =~/(\S+)/g; }

	if ($opt{with_id}) {
		my $id = shift @i;
		push @ids, $id;
	}

	# select variables according to $opt{var}
	my @j;
	if (@headers >0 and $opt{var}) {
		for (my $k=0; $k <@i; $k++) {
			my $header = $headers[$k];
			if (defined $want_vars->{$header}) {
				push @j, $i[$k];
			}
		}
		@i = @j;
	}



	if (not defined $num_vars) { $num_vars = @i-1; }
	elsif ($num_vars != @i-1) { 
		die "ERROR:Following line has a different number of variables than other lines!\n$line\n";
	}

	push @data, \@i;
}
close FH;

my $num_samples = scalar @data;
my @info;

for (my $i=0; $i < $num_vars; $i++) {
	normalize(\@data, $i, $num_samples, \@info, $opt{missing});
}

convert_outcomes(\@data, $num_vars, $num_samples, \@info, $opt{missing});

my $i= -1;
open OUT, ">".$opt{output} or die "ERROR:Cannot open output file!\n";
foreach my $row (@data) {
	$i++;
	print OUT $ids[$i]."\t" if $opt{with_id};
	print OUT join "\t", @$row;
	print OUT "\n";
}
close OUT;

open OUT, ">".$opt{output}.".info";
foreach my $line (@info) { print OUT $line; }
close OUT;

#-----------------------------------------------------------------------------
#----------------------------------- SUBS-------------------------------------
#-----------------------------------------------------------------------------
sub normalize {
	my ($data, $col_index, $num_samples, $info, $missing) = @_;

	convert_categorical_values($data, $col_index, $num_samples, $info, $missing);
	convert_missing_values($data, $col_index, $num_samples, $info, $missing);	
	normalize_01($data, $col_index, $num_samples, $info, $missing);
}

#-----------------------------------------------------------------------------
sub normalize_01 {
	my ($data, $col_index, $num_samples, $info, $missing)  = @_;

	my ($min, $max);

	my $col = $col_index +1;
	for (my $i=0; $i < $num_samples; $i++) {
		my $val = $data->[$i][$col_index];

		if (not defined $min or $val <$min) { $min = $val; }
		if (not defined $max or $val >$max) { $max = $val; }
	}

	if ($min>=0 and $max <=1) { return; }

	for (my $i=0; $i < $num_samples; $i++) {
		my $val = $data->[$i][$col_index];
	
		my $new_val;
		if ($min==$max) { $new_val = 0;}
		else {
			$new_val = ($val -$min) / ($max-$min);
		}
		$data->[$i][$col_index] = $new_val;
	}

	push @$info, "normalize\tcol:$col\tmin:$min\tmax:$max\n";
}

#-----------------------------------------------------------------------------
sub convert_missing_values {
	my ($data, $col_index, $num_samples, $info, $missing)  = @_;

	my $col = $col_index +1;
	my $sum = 0;
	my $count = 0;

	for (my $i=0; $i < $num_samples; $i++) {
		my $val = $data->[$i][$col_index];

		next if $val eq $missing;

		$count ++;
		$sum += $val;
	}
	my $ave = $sum /$count;

	my $converted = 0;
	for (my $i=0; $i < $num_samples; $i++) {
		if ($data->[$i][$col_index] eq $missing) {
			$data->[$i][$col_index] = $ave;
			$converted ++;
		}
	}

	push @$info, "count_missing_values\tcol:$col\t$converted\n" if $converted;
}
		
#-----------------------------------------------------------------------------
sub convert_outcomes {
	my ($data, $col_index, $num_samples, $info, $missing)  = @_;

	my $col = $col_index +1;
	my %keys;

	for (my $i=0; $i < $num_samples; $i++) {

		if ($data->[$i][$col_index] =~/^-?(0|([1-9][0-9]*))(\.[0-9]+)?([eE][-+]?[0-9]+)?$/){
			$data->[$i][$col_index] *=1;
		}
		my $val = $data->[$i][$col_index];

		$keys{$val} = 1;
	}

	foreach my $key (%keys) {
		if ($key ne $missing and $key ne 0 and $key ne 1) {
			die "The outcome (last column) must be 0, 1 or the missing string.\n";
		}
	}
}

#-----------------------------------------------------------------------------
sub convert_categorical_values {
	my ($data, $col_index, $num_samples, $info, $missing)  = @_;

	my $col = $col_index +1;
	my %keys;

	my $has_nonnumber = 0;
	for (my $i=0; $i < $num_samples; $i++) {
		my $val = $data->[$i][$col_index];

		next if $val eq $missing;
		$keys{$val} = 1;
		if (!is_a_number($val)) { $has_nonnumber = 1; }
	}

	if ($has_nonnumber) {
		if (keys %keys >2) {
			die "ERROR:Column $col has more than 2 categorical values. Please fix.\n";
		} else {
			my @keys = keys %keys;
			push @$info, "recoding\tcol:$col\t$keys[0]\t0\n";
			push @$info, "recoding\tcol:$col\t$keys[1]\t1\n" if defined $keys[1];

			for (my $i=0; $i < $num_samples ; $i++) {
				next if $missing eq  $data->[$i][$col_index];

				if ($data->[$i][$col_index] eq $keys[0]) {
					$data->[$i][$col_index] = 0;
				} else {
					$data->[$i][$col_index] = 1;
				}
			}
		}
	}
}

#-----------------------------------------------------------------------------
sub is_a_number {
	my $x = shift;

	if ($x =~/e/i) {
		my @tmp = $x =~/(e)/gi;
		if (@tmp >1) { return 0; }
		else {
			my ($a, $b) = split /[Ee]/, $x;
			return is_a_number($a) and is_a_number($b);
		}
	} else {
		$x =~s/^(\+|\-)//;

		if ($x =~/\./) {
			my @tmp = $x =~/(\.)/g;
			if (@tmp >1) { return 0; }

			my ($a, $b) = split /\./, $x;

			if ($a =~/^\d*$/ and $b =~/^\d*$/ and !($a eq "" and $b eq "") ) { return 1; }
			else { return 0; }
		}
		else {
			return $x =~ /^\d+$/;
		}
	}
}

	
#-----------------------------------------------------------------------------
sub parse_opt_var {
	my $string = shift;

	my %vars;
	if (-e $string) {
		open(my $fh, $string);
		while (my $line = <$fh>) {
			my @i = $line =~/(\S+)/g;
			foreach my $v (@i) { $vars{$v} = 1; }
		}
		close $fh;
	} else {
		foreach my $v (split /,/, $string) {
			$vars{$v} = 1;
		}
	}
	return \%vars;
}

#-----------------------------------------------------------------------------

