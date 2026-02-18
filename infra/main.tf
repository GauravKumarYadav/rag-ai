provider "oci" {}

resource "oci_core_instance" "generated_oci_core_instance" {
	agent_config {
		is_management_disabled = "false"
		is_monitoring_disabled = "false"
		plugins_config {
			desired_state = "DISABLED"
			name = "Vulnerability Scanning"
		}
		plugins_config {
			desired_state = "DISABLED"
			name = "Management Agent"
		}
		plugins_config {
			desired_state = "ENABLED"
			name = "Custom Logs Monitoring"
		}
		plugins_config {
			desired_state = "DISABLED"
			name = "Compute RDMA GPU Monitoring"
		}
		plugins_config {
			desired_state = "ENABLED"
			name = "Compute Instance Monitoring"
		}
		plugins_config {
			desired_state = "DISABLED"
			name = "Compute HPC RDMA Auto-Configuration"
		}
		plugins_config {
			desired_state = "DISABLED"
			name = "Compute HPC RDMA Authentication"
		}
		plugins_config {
			desired_state = "ENABLED"
			name = "Cloud Guard Workload Protection"
		}
		plugins_config {
			desired_state = "DISABLED"
			name = "Block Volume Management"
		}
		plugins_config {
			desired_state = "DISABLED"
			name = "Bastion"
		}
	}
	availability_config {
		recovery_action = "RESTORE_INSTANCE"
	}
	availability_domain = "oVIz:AP-HYDERABAD-1-AD-1"
	compartment_id = "ocid1.tenancy.oc1..aaaaaaaab3fn2mfeejtfbvwoeh2rfq2mepuy3mrrzbqwnj6pgcrdkrs7vzkq"
	create_vnic_details {
		assign_ipv6ip = "false"
		assign_private_dns_record = "true"
		assign_public_ip = "true"
		subnet_id = "ocid1.subnet.oc1.ap-hyderabad-1.aaaaaaaakprgnssu5m7mysi45zaoi5zbn4vmtgcrt4kuyq6fjlfj7wsx57ga"
	}
	display_name = "instance-20260218-2223"
	instance_options {
		are_legacy_imds_endpoints_disabled = "false"
	}
	is_pv_encryption_in_transit_enabled = "true"
	metadata = {
		"ssh_authorized_keys" = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCO3/oyIfuMJbanXAp9zMQWHw7gSIQobO2XDD3IoeDwa0deK5fNAv9ERxEGxhHJAxvcHErIHXViyjQqQOvVFi6pl3YHhcQdx5cZvrMb2rRCaC169x49xzq3FYDLQU0STrrSQEa3MttO1Uuhou+rgVuCEDqmFWumVBQacezLl4BneKT8eLUedNM4Zy9ANk5l0XvLYYbFw2r22HNG4TiP1JKitVZ0dap2WZ34KrCyfz3HrSNJzwfF65COFWRbERWiFdVoL0K7YBCPp/noCjYI+MQ9XiJ/+XvSsyxvb46J8NAvWcmb5h61dHeHaeubq2qq8wXYygNxJVV1fIbVtSF1tG9j ssh-key-2026-02-18"
	}
	shape = "VM.Standard.A1.Flex"
	shape_config {
		memory_in_gbs = "24"
		ocpus = "4"
	}
	source_details {
		boot_volume_size_in_gbs = "100"
		boot_volume_vpus_per_gb = "10"
		source_id = "ocid1.image.oc1.ap-hyderabad-1.aaaaaaaape6pxqtgswnbolzxyjok2klghymdar5hqa2qnu3mvr2tsldahxxq"
		source_type = "image"
	}
}
