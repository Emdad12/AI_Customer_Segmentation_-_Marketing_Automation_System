def generate_email(segment):

    templates = {
    "Churned / Lost Customers":
    """
    Subject: We Miss You!

    It's been a while since your last purchase.
    Come back and enjoy a special welcome-back offer.
    """,

    "VIP / Champions":
    """
    Subject: Exclusive VIP Reward

    Thank you for being one of our most valued customers.
    Enjoy exclusive rewards and premium offers.
    """,

    "Inactive Discount Seekers":
    """
    Subject: Flash Sale Alert

    Hi Mr/Mrs, since you love a great deal, we're giving you
    early access to our 60% off clearance event. Shop now before it's gone!
    """,

    "Active Regulars":
    """
    Subject: New Arrivals Just for You

    Hi, we noticed you’ve been shopping with us lately! Based on your
    favorites, we think you'll love these new arrivals.
    """
    }

    return templates.get(segment, "Thank you for using our service.")